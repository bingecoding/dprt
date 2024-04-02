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

#ifndef VPLCPU_H
#define VPLCPU_H

#include <vector>

#include "engines/renderengine.h"
#include "sampler.h"
#include "pointcloud.h"
#include "engines/gputypes.h"
#include "lighttree.h"
#include "priorityqueue.h"

#include "openclintersectiondevice.h"

#include <boost/thread/thread.hpp>

using namespace std;

class VplCpuRenderEngine;

//------------------------------------------------------------------------------
// VPL Integrator
//------------------------------------------------------------------------------

class VplIntegrator {
public:
    VplIntegrator(VplCpuRenderEngine *renderEngine, Sampler *samp);
    ~VplIntegrator();
    
    void Li();
    
    Spectrum GetMaterialTerm(Cluster *node, Point shadingPoint,
                             Normal shadingNormal,
                             const SurfaceMaterial *triSurfMat, const Vector wo);
    float GetVisibilityTerm(Cluster *node, Point shadingPoint);
    float GetGeometricTerm(Cluster *node, Point shadingPoint);
    
    float MaxCosine(Cluster *node, Point shadingPoint,
                    Normal shadingNormal);
    
    float GetBoundGeometricTerm(Cluster *node, Point &shadingPoint, Normal &shadingNormal);
    
    Spectrum GetBoundMaterialTerm(Cluster *node, Point shadingPoint,
                                  Normal shadingNormal,
                                  const SurfaceMaterial *triSurfMat,
                                  const Vector wo);
    
    void GetClusterRadiance(Cluster *node, Point shadingPoint,
                            Normal shadingNormal,
                            const SurfaceMaterial *triSurfMat, const Vector wo);
    
    
    double m_statsRenderingStart;
    double m_statsTotalSampleCount;
    
    void LightCuts();
    
private:
    
    VplCpuRenderEngine *m_renderEngine;
    Sampler *m_sampler;
    RandomGenerator *m_rnd;
    
    Scene *m_scene;
    SampleBuffer *m_sampleBuffer;
    
    PointCloud *m_pointCloud;

    struct CompareErrorBound {
        
        bool operator()(const Cluster *c1, const Cluster *c2) const {
            // The biggest error bound value will be at the front of the heap
            return c1->errorBound.Y() < c2->errorBound.Y();
        }
        
    };
    
};


class VplCpuRenderThread {
public:
    VplCpuRenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                           const unsigned int samplePerPixel, VplCpuRenderEngine *renderEngine);
    /*
    ~VplCpuRenderThread();
    */
    void Start();
    void Interrupt();
    void Stop();
    
    //unsigned int GetPass() const { return sampler->GetPass(); }
    
    friend class PathGPURenderEngine;
    
protected:
    unsigned int m_threadIndex;
    VplCpuRenderEngine *m_renderEngine;
    
private:
    static void RenderThreadImpl(VplCpuRenderThread *renderThread);
    
    float m_samplingStart;
    Sampler *m_sampler;
    VplIntegrator *m_vplIntegrator;
    //RayBuffer *rayBuffer;
    
    boost::thread *m_renderThread;
};

//------------------------------------------------------------------------------
// Path Tracing CPU-only render engine
//------------------------------------------------------------------------------

class VplCpuRenderEngine : public RenderEngine {
public:
    VplCpuRenderEngine(Scene *scn, Film *flm, boost::mutex *filmMutex,
                      const Properties &cfg, bool lightCuts=false);
    virtual ~VplCpuRenderEngine();
    
    
    void Start();
    
    void Interrupt();
    
    void Stop();
    
    
    unsigned int GetPass() const;
    unsigned int GetThreadCount() const;
    RenderEngineType GetEngineType() const { return VPLCPU; }
    
    double GetTotalSamplesSec() const {
        return (m_elapsedTime == 0.0) ? 0.0 : (m_samplesCount / m_elapsedTime);
    }
    double GetRenderingTime() const { return (m_startTime == 0.0) ? 0.0 : (WallClockTime() - m_startTime); }

    void UpdateFilm();
    
    friend class VplCpuRenderThread;
    friend class VplIntegrator;
    
    unsigned int m_samplePerPixel;
    
    // Signed because of the delta parameter
    int m_maxPathDepth;
    
    int m_rrDepth;
    float m_rrImportanceCap;
    float m_clamping;
    
    int m_lightTreeCutSize;
    
private:
    vector<VplCpuRenderThread *> m_renderThreads;
    SampleBuffer *m_sampleBuffer;
    
    double m_screenRefreshInterval; // in seconds
    
    double m_startTime;
    double m_elapsedTime;
    unsigned long long m_samplesCount;
    
    bool m_lightCutsMode;
    
    bool m_enableOpenGLInterop, m_dynamicCamera;
};

#endif
