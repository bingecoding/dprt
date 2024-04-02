/***************************************************************************
 * Copyright (c) 2014 Michael Walch                                        *
 *                                                                         *
 *   This project is based on LuxRays ; see https://luxcorerender.org       *
 *   LuxRays is the part of LuxRender dedicated to accelerate the          *
 *   ray intersection process by using GPUs.                               *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#include "engines/reconstructioncutsgpu.h"


//------------------------------------------------------------------------------
// ReconstructionCutsGpuRenderEngine
//------------------------------------------------------------------------------

ReconstructionCutsGpuRenderEngine::ReconstructionCutsGpuRenderEngine(Scene *scene, Film *film,
                                       boost::mutex *filmMutex,
                                       vector<OpenCLIntersectionDevice *>
                                       intersectionDevices,
                                       const Properties &cfg) :
RenderEngine(scene, film, filmMutex, cfg)
{
    m_samplePerPixel = max(1, cfg.GetInt("path.sampler.spp"));
    m_samplePerPixel *=m_samplePerPixel;
    m_maxPathDepth = cfg.GetInt("path.maxdepth");
    m_rrDepth = cfg.GetInt("path.russianroulette.depth");
    m_rrImportanceCap = cfg.GetFloat("path.russianroulette.cap");
    
    m_clamping = cfg.GetFloat("vpl.clamping");
    
    m_globalWorkGroupSize = cfg.GetInt("opencl.global.work.size");
    m_lightTreeCutSize = cfg.GetInt("light.tree.cut.size");
    
	m_startTime = 0.0;
	m_samplesCount = 0;
    
    m_sampleBuffer = m_film->GetFreeSampleBuffer();
    
    // Look for OpenCL devices
	for (size_t i = 0; i < intersectionDevices.size(); ++i) {
        m_oclIntersectionDevices.push_back((OpenCLIntersectionDevice *)
                                           intersectionDevices[i]);
	}
    
	cerr << "Found "<< m_oclIntersectionDevices.size() <<
    " OpenCL intersection devices for ReconstructionCutsGPU render engine" << endl;
    
    if (m_oclIntersectionDevices.size() < 1)
		throw runtime_error("Unable to find an OpenCL intersection device for\
                            ReconstructionCutsGPU render engine");
    
    const unsigned int seedBase = (unsigned int)(WallClockTime() / 1000.0);
    
	// Create and start render threads
	const size_t renderThreadCount = m_oclIntersectionDevices.size();
	cerr << "Starting "<< renderThreadCount << " ReconstructionCutsGPU render threads"
    << endl;
	
    for (size_t i = 0; i < renderThreadCount; ++i) {
		ReconstructionCutsGpuRenderThread *t = new ReconstructionCutsGpuRenderThread(
                                                     i,
                                                     seedBase + i * m_globalWorkGroupSize,
                                                     i /(float)renderThreadCount,
                                                     m_oclIntersectionDevices[i],
                                                       this);
		m_renderThreads.push_back(t);
	}
    
}

ReconstructionCutsGpuRenderEngine::~ReconstructionCutsGpuRenderEngine() {
	if (m_started) {
		Interrupt();
		Stop();
	}
    
	for (size_t i = 0; i < m_renderThreads.size(); ++i)
		delete m_renderThreads[i];
    
}

void ReconstructionCutsGpuRenderEngine::Start() {
	RenderEngine::Start();
    
	m_samplesCount = 0;
	m_elapsedTime = 0.0f;
    
	for (size_t i = 0; i < m_renderThreads.size(); ++i)
		m_renderThreads[i]->Start();
    
	m_startTime = WallClockTime();
}

void ReconstructionCutsGpuRenderEngine::Interrupt() {
	for (size_t i = 0; i < m_renderThreads.size(); ++i)
		m_renderThreads[i]->Interrupt();
}

void ReconstructionCutsGpuRenderEngine::Stop() {
	RenderEngine::Stop();
    
	for (size_t i = 0; i < m_renderThreads.size(); ++i)
		m_renderThreads[i]->Stop();
    
	UpdateFilm();
}

unsigned int ReconstructionCutsGpuRenderEngine::GetThreadCount() const {
	return m_renderThreads.size();
}

void ReconstructionCutsGpuRenderEngine::UpdateFilm()
{
    boost::unique_lock<boost::mutex> lock(*m_filmMutex);
    
    m_elapsedTime = WallClockTime() - m_startTime;
	const unsigned int imgWidth = m_film->GetWidth();
	const unsigned int pixelCount = imgWidth * m_film->GetHeight();
    
    m_film->Reset();
    
    unsigned long long totalCount = 0;
	for (unsigned int p = 0; p < pixelCount; ++p) {
		Spectrum c;
		unsigned int count = 0;
		for (size_t i = 0; i < m_renderThreads.size(); ++i) {
			c += m_renderThreads[i]->m_frameBuffer[p].c;
			count += m_renderThreads[i]->m_frameBuffer[p].count;
		}
        
		if (count > 0) {
			const float scrX = p % imgWidth;
			const float scrY = p / imgWidth;
			c /= count;
			m_sampleBuffer->SplatSample(scrX, scrY, c);
            
			if (m_sampleBuffer->IsFull()) {
				// Splat all samples on the film
				m_film->SplatSampleBuffer(true, m_sampleBuffer);
				m_sampleBuffer = m_film->GetFreeSampleBuffer();
			}
            
			totalCount += count;
		}
        
		if (m_sampleBuffer->GetSampleCount() > 0) {
			// Splat all samples on the film
			m_film->SplatSampleBuffer(true, m_sampleBuffer);
			m_sampleBuffer = m_film->GetFreeSampleBuffer();
		}
	}
    
	m_samplesCount = totalCount;
}

unsigned int ReconstructionCutsGpuRenderEngine::GetPass() const {
	return m_samplesCount / (m_film->GetWidth() * m_film->GetHeight());
}

