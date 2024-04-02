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

#ifndef RECONSTRUCTIONCUTSGPU_H
#define RECONSTRUCTIONCUTSGPU_H

#include <vector>

#include "engines/renderengine.h"
#include "engines/gputypes.h"
#include "openclintersectiondevice.h"
#include "lighttree.h"

#include <boost/thread/thread.hpp>

using namespace std;

class ReconstructionCutsGpuRenderEngine;

//------------------------------------------------------------------------------
// Path Tracing GPU-only render threads
//------------------------------------------------------------------------------

class ReconstructionCutsGpuRenderThread {
public:
	ReconstructionCutsGpuRenderThread(const unsigned int index, const unsigned int seedBase,
                        const float samplingStart, OpenCLIntersectionDevice *device,
                        ReconstructionCutsGpuRenderEngine *re);
	~ReconstructionCutsGpuRenderThread();
    
	void Start();
    void Interrupt();
	void Stop();
    
	friend class ReconstructionCutsGpuRenderEngine;
    
private:
	static void RenderThreadImpl(ReconstructionCutsGpuRenderThread *renderThread);
    
	void InitRender();
	void EnqueueInitFrameBufferKernel(const bool clearPBO = false);
    
	OpenCLIntersectionDevice *m_intersectionDevice;
    
	// OpenCL variables
    cl::Kernel *m_integratorKernel;
    
	string m_kernelsParameters;
	cl::Kernel *m_initKernel;
	size_t m_initWorkGroupSize;
	cl::Kernel *m_initFBKernel;
	size_t m_initFBWorkGroupSize;
	cl::Kernel *m_updatePBKernel;
	size_t m_updatePBWorkGroupSize;
	cl::Kernel *m_updatePBBluredKernel;
	size_t m_updatePBBluredWorkGroupSize;
	cl::Kernel *m_advancePathStep1Kernel;
	size_t m_advancePathStep1WorkGroupSize;
	cl::Kernel *m_advancePathStep2Kernel;
	size_t m_advancePathStep2WorkGroupSize;
    cl::Kernel *m_resetHashTableKernel;

    cl::Buffer *m_lightTreeBuff;
    cl::Buffer *m_lightCutsHeapBuff;
    cl::Buffer *m_heapIndexBuff;
    cl::Buffer *m_hashTableBuff;
    cl::Buffer *m_standardLightCutsBuff;
    cl::Buffer *m_sampleCutsBuff;
    cl::Buffer *m_sampleStackBuff;
    
	cl::Buffer *m_raysBuff;
	cl::Buffer *m_hitsBuff;
    cl::Buffer *m_raysClusterBuff;
    cl::Buffer *m_hitsClusterBuff;
    
	cl::Buffer *m_pathsBuff;
	cl::Buffer *m_frameBufferBuff;
	cl::Buffer *m_materialsBuff;
	cl::Buffer *m_meshIDBuff;
	cl::Buffer *m_meshMatsBuff;
	cl::Buffer *m_infiniteLightBuff;
	cl::Buffer *m_normalsBuff;
	cl::Buffer *m_trianglesBuff;
	cl::Buffer *m_colorsBuff;
	cl::Buffer *m_cameraBuff;
	cl::Buffer *m_triLightsBuff;
    
    cl::Buffer *m_texMapRGBBuff;
    cl::Buffer *m_texMapAlphaBuff;
    cl::Buffer *m_texMapDescBuff;
    cl::Buffer *m_meshTexsBuff;
    cl::Buffer *m_uvsBuff;
    
	double m_lastCameraUpdate;
    
	float m_samplingStart;
    
	boost::thread *m_renderThread;
    
	unsigned int m_threadIndex;
	ReconstructionCutsGpuRenderEngine *m_renderEngine;
	PixelGPU *m_frameBuffer;
    
	bool m_started, m_reportedPermissionError;

    float m_sceneDiag;
};

//------------------------------------------------------------------------------
// Path Tracing GPU-only render engine
//------------------------------------------------------------------------------

class ReconstructionCutsGpuRenderEngine : public RenderEngine {
public:
	ReconstructionCutsGpuRenderEngine(Scene *scn, Film *flm, boost::mutex *filmMutex,
                        vector<OpenCLIntersectionDevice *> intersectionDevices, const Properties &cfg);
	virtual ~ReconstructionCutsGpuRenderEngine();
    
	void Start();
	void Interrupt();
	void Stop();
    
	unsigned int GetPass() const;
	unsigned int GetThreadCount() const;
	RenderEngineType GetEngineType() const { return RECONSTRUCTIONCUTSGPU; }
    
	double GetTotalSamplesSec() const {
		return (m_elapsedTime == 0.0) ? 0.0 : (m_samplesCount / m_elapsedTime);
	}
	double GetRenderingTime() const { return (m_startTime == 0.0) ? 0.0 : (WallClockTime() - m_startTime); }
    unsigned long long GetSamplesCount() const { return m_samplesCount; }
    
    void UpdateFilm();
    
	friend class ReconstructionCutsGpuRenderThread;
    
	unsigned int m_samplePerPixel;
    
	// Signed because of the delta parameter
	int m_maxPathDepth;
    float m_clamping;
	int m_rrDepth;
	float m_rrImportanceCap;
    
    int m_globalWorkGroupSize;
    int m_lightTreeCutSize;
    
private:
	vector<OpenCLIntersectionDevice *> m_oclIntersectionDevices;
	vector<ReconstructionCutsGpuRenderThread *> m_renderThreads;
	SampleBuffer *m_sampleBuffer;
    
	double m_screenRefreshInterval; // in seconds
    
	double m_startTime;
	double m_elapsedTime;
	unsigned long long m_samplesCount;
    
	bool m_dynamicCamera;
    
};


#endif
