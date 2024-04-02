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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdexcept>

#include <boost/thread/mutex.hpp>

#include "engines/pathcpu.h"
#include "raytracer.h"
#include "bvhaccel.h"
#include "buffer.h"
#include "sampler.h"



//------------------------------------------------------------------------------
// PathCPURenderEngine
//------------------------------------------------------------------------------

PathCPURenderEngine::PathCPURenderEngine(Scene *scene, Film *film,
                                         boost::mutex *filmMutex,
                                         const Properties &cfg) :
RenderEngine(scene, film, filmMutex, cfg)
{
    m_samplePerPixel = max(1, cfg.GetInt("path.sampler.spp"));
    m_samplePerPixel *=m_samplePerPixel;
    m_maxPathDepth = cfg.GetInt("path.maxdepth");
    m_rrDepth = cfg.GetInt("path.russianroulette.depth");
    m_rrImportanceCap = cfg.GetFloat("path.russianroulette.cap");
    
    m_startTime = 0.0;
    m_samplesCount = 0;
    
    m_sampleBuffer = m_film->GetFreeSampleBuffer();
    
    const unsigned int seedBase = (unsigned int)(WallClockTime() / 1000.0);
    
    // Create and start render threads
    const size_t renderThreadCount = 1;
    cerr << "Starting "<< renderThreadCount << " PathCPU render threads"
    << endl;
    
    for (size_t i = 0; i < renderThreadCount; ++i) {
        PathCPURenderThread *t = new PathCPURenderThread(i,
                                                         seedBase,
                                                         i /(float)renderThreadCount,
                                                         m_samplePerPixel,
                                                         this);
        m_renderThreads.push_back(t);
    }
    
}

void PathCPURenderEngine::Start()
{
    RenderEngine::Start();
    
    m_samplesCount = 0;
    m_elapsedTime = 0.0f;
    
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Start();
    
    m_startTime = WallClockTime();
}

void PathCPURenderEngine::Interrupt()
{
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Interrupt();
}

PathCPURenderEngine::~PathCPURenderEngine()
{
}

unsigned int PathCPURenderEngine::GetPass() const {
    return m_samplesCount / (m_film->GetWidth() * m_film->GetHeight());
}

unsigned int PathCPURenderEngine::GetThreadCount() const {
    return m_renderThreads.size();
}

//------------------------------------------------------------------------------
// PathCPURenderThread
//------------------------------------------------------------------------------

PathCPURenderThread::PathCPURenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                                         const unsigned int samplePerPixel, PathCPURenderEngine *renderEngine)
{
    m_renderEngine = renderEngine;
    m_samplingStart = samplingStart;
    m_threadIndex = index;
    
    const unsigned int startLine = Clamp<unsigned int>(
                                                       m_renderEngine->m_film->GetHeight() * m_samplingStart,
                                                       0, m_renderEngine->m_film->GetHeight() - 1);
    
    m_sampler = new RandomSampler(false, seedBase + m_threadIndex + 1,
                                m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(),
                                samplePerPixel, startLine);

    m_pathIntegrator = new PathIntegrator(m_renderEngine, m_sampler);
    
}

void PathCPURenderThread::Start()
{

    
    const unsigned int startLine = Clamp<unsigned int>(
                                                       m_renderEngine->m_film->GetHeight() * m_samplingStart,
                                                       0, m_renderEngine->m_film->GetHeight() - 1);
     
    m_sampler->Init(m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(), startLine);
    
    // Create the thread for the rendering
    m_renderThread = new boost::thread(boost::bind(PathCPURenderThread::RenderThreadImpl, this));
}

void PathCPURenderThread::Interrupt() {
    if (m_renderThread)
        m_renderThread->interrupt();
}

void PathCPURenderThread::RenderThreadImpl(PathCPURenderThread *renderThread) {
    cerr << "[PathCPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread started" << endl;
    
    try {
        PathIntegrator *pathIntegrator = renderThread->m_pathIntegrator;
        while (!boost::this_thread::interruption_requested()) {
            pathIntegrator->Li();
        }
        
        cerr << "[PathCPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    } catch (boost::thread_interrupted) {
        cerr << "[PathCPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    }

}

//------------------------------------------------------------------------------
// Path Integrator
//------------------------------------------------------------------------------
PathIntegrator::PathIntegrator(PathCPURenderEngine *re, Sampler *samp) :
m_renderEngine(re), m_sampler(samp)
{
    m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
    m_statsRenderingStart = WallClockTime();
    m_statsTotalSampleCount = 0;
    unsigned long seedBase = (unsigned long)(WallClockTime() / 1000.0);
    
    m_rndGen = new RandomGenerator(seedBase);

}


void PathIntegrator::Li()
{
    Sample sample;
    
    
    const unsigned int width = m_renderEngine->m_film->GetWidth();
    const unsigned int height = m_renderEngine->m_film->GetHeight();
    
    const unsigned int pixelCount = width * height;
   
    Scene *scene = m_renderEngine->m_scene;
    for (unsigned int i = 0; i < pixelCount; ++i) {
    
        Ray pathRay;
        RayHit rayHit;
        m_sampler->GetNextSample(&sample);
        m_renderEngine->m_scene->m_camera->GenerateRay(
                                                       sample.screenX, sample.screenY,
                                                       m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(), &pathRay,
                                                       m_sampler->GetLazyValue(&sample), m_sampler->GetLazyValue(&sample), m_sampler->GetLazyValue(&sample));
        
        // Generate rays from camera
        /*
        const unsigned int x = i % width;
        const unsigned int y = i / width;
        const float scrX = x + m_rndGen->floatValue() - 0.5f;
        const float scrY = y + m_rndGen->floatValue() - 0.5f;
        m_renderEngine->m_scene->m_camera->GenerateRay(scrX, scrY, width, height, &pathRay,
                                       m_rndGen->floatValue(), m_rndGen->floatValue(),
                                       m_rndGen->floatValue());
        */
        const float scrX = sample.screenX;
        const float scrY = sample.screenY;
        
        Spectrum pathThroughput(1.f, 1.f, 1.f);
        Spectrum radiance(0.f, 0.f, 0.f);
        bool specular = false;
        
        for (int bounces = 0; ; ++bounces) {
            
            if(scene->Intersect(&pathRay, &rayHit)) {
                
                const unsigned int currentTriangleIndex = rayHit.index;
                const unsigned int currentMeshIndex =  m_renderEngine->m_scene->GetAccelerator()->GetMeshID(currentTriangleIndex);
                
                // Get the triangle
                const TriangleMesh *mesh = m_renderEngine->m_scene->m_objectMeshes[currentMeshIndex];
                const unsigned int triIndex = m_renderEngine->m_scene->GetAccelerator()->GetMeshTriangleID(currentTriangleIndex);
                
                // Get the material
                const Material *triMat = m_renderEngine->m_scene->m_objectMaterials[currentMeshIndex];
                
                if((triMat->IsLightSource())) {
                    const LightMaterial *matLight = (LightMaterial *)triMat;
                    Spectrum Le = matLight->Le(mesh, triIndex, -pathRay.d);
                    if(bounces == 0  || specular) {
                        radiance += Le * pathThroughput;
                        /*if(matLight->GetType() == AREALIGHT) {
                            TriangleLight *triLight = (TriangleLight *)scene->GetLightSource(triIndex);
                            float lightPdf;
                            triLight->Sample_L(scene,m_rndGen->floatValue(), m_rndGen->floatValue(),
                                               m_rndGen->floatValue(), m_rndGen->floatValue(),
                                               m_rndGen->floatValue(), &lightPdf, &pathRay);
                            m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                            continue;
                        } This does not seem to make a difference, i.e. continuing
                          the path if the first bounce hits a light source.
                         */
                        
                    }

                    // Terminate the path
                    m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                    break;
                }
                
                //--------------------------------------------------------------
                // Build the shadow rays (if required)
                //--------------------------------------------------------------
                
                // Interpolate face normal
                Normal N = mesh->InterpolateTriNormal(triIndex, rayHit.b1, rayHit.b2);
                
                const SurfaceMaterial *triSurfMat = (SurfaceMaterial *) triMat;
                const Point hitPoint = pathRay(rayHit.t);
                const Vector wo = -pathRay.d;
                
                Spectrum surfaceColor;
                if (mesh->HasColors())
                    surfaceColor = mesh->InterpolateTriColor(triIndex, rayHit.b1, rayHit.b2);
                else
                    surfaceColor = Spectrum(1.f, 1.f, 1.f);
                
                // Flip the normal if required
                Normal shadeN = (Dot(pathRay.d, N) > 0.f) ? -N : N;
                
                Spectrum lightColor(0.f, 0.f, 0.f);
                if (triSurfMat->IsDiffuse()) {
                    
                    // ONE UNIFORM direct sampling light strategy
                    const Spectrum lightThroughtput = pathThroughput * surfaceColor;
                    // Select the light to sample
                    float lightStrategyPdf;
                    const LightSource *light =  m_renderEngine->m_scene->SampleAllLights(m_rndGen->floatValue(),
                                                                      &lightStrategyPdf);
                    
                    Ray shadowRay;
                    float lightPdf;
                    Spectrum Li = light->Sample_L(m_renderEngine->m_scene, hitPoint, &shadeN,
                                    m_rndGen->floatValue(), m_rndGen->floatValue(), m_rndGen->floatValue(),
                                    &lightPdf, &shadowRay);
                    
                    if (lightPdf > 0.f && !Li.Black()) {
                    
                        const Vector lwi = shadowRay.d;
                        Spectrum f = triSurfMat->f(wo, lwi, shadeN);
                        bool visible = !scene->IntersectP(&shadowRay);
                        if (!f.Black() && visible) {
                            radiance += f * Li * lightThroughtput * Dot(shadeN, lwi) / (lightPdf * lightStrategyPdf);
                        }
              
                    }
                    
                }
                
                //--------------------------------------------------------------
                // Build the next vertex path ray
                //--------------------------------------------------------------
                
                float fPdf;
                Vector wi;
                bool specularBounce = false;
                const Spectrum f = triSurfMat->Sample_f(wo, &wi, N, shadeN,
                                                        m_rndGen->floatValue(), m_rndGen->floatValue(), m_rndGen->floatValue(),
                                                        false, &fPdf, specularBounce);// * surfaceColor;
                specular = triSurfMat->IsSpecular();
                
                if ((fPdf <= 0.f) || f.Black()  ) {
                    // Terminate the path
                    m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                    break;
                }
                
                pathThroughput *= f / fPdf;
                
                // Possibly terminate the path
                if (bounces > 3) {
                    float continueProbability = min(.5f, pathThroughput.Y());
                    if(m_rndGen->floatValue() > continueProbability) {
                        // Terminate the path
                        m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                        break;
                    }
                    pathThroughput /= continueProbability;
                }
                
                if (bounces == m_renderEngine->m_rrDepth) {
                    // Terminate the path
                    m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                    break;
                }
                
                // Continue path
                pathRay.o = hitPoint;
                pathRay.d = wi;

            }
            else {
                break;
            }
            
        }
        
        // Check if the sample buffer is full
        if (m_sampleBuffer->IsFull()) {
            m_statsTotalSampleCount += m_sampleBuffer->GetSampleCount();
            
            // Splat all samples on the film
            m_renderEngine->m_film->SplatSampleBuffer(m_sampler->IsPreviewOver(), m_sampleBuffer);
            m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
        }

    }

}

