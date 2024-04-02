/***************************************************************************
 * Copyright (c) 2014 Michael Walch                                        *
 *                                                                         *
 *   This project is based on LuxRays; see https://luxcorerender.org       *
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

#ifndef PATHCPU_H
#define PATHCPU_H

#include <vector>

#include "engines/renderengine.h"
#include "sampler.h"

#include "openclintersectiondevice.h"

#include <boost/thread/thread.hpp>

using namespace std;

class PathCPURenderEngine;

//------------------------------------------------------------------------------
// Path Integrator
//------------------------------------------------------------------------------

class PathIntegrator {
public:
    PathIntegrator(PathCPURenderEngine *renderEngine, Sampler *samp);
    ~PathIntegrator();
    
    void Li();
    
    double m_statsRenderingStart;
    double m_statsTotalSampleCount;
    
    private:
    PathCPURenderEngine *m_renderEngine;
    Sampler *m_sampler;
    RandomGenerator *m_rndGen;
    
    SampleBuffer *m_sampleBuffer;
    
};


class PathCPURenderThread {
public:
    PathCPURenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                           const unsigned int samplePerPixel, PathCPURenderEngine *renderEngine);
    /*
    ~PathCPURenderThread();
    */
    void Start();
    void Interrupt();
    //void Stop();
    
    //unsigned int GetPass() const { return sampler->GetPass(); }
    
    friend class PathGPURenderEngine;
    
protected:
    unsigned int m_threadIndex;
    PathCPURenderEngine *m_renderEngine;
    
private:
    static void RenderThreadImpl(PathCPURenderThread *renderThread);
    
    float m_samplingStart;
    Sampler *m_sampler;
    PathIntegrator *m_pathIntegrator;
    //RayBuffer *rayBuffer;
    
    boost::thread *m_renderThread;
};

//------------------------------------------------------------------------------
// Path Tracing CPU-only render engine
//------------------------------------------------------------------------------

class PathCPURenderEngine : public RenderEngine {
public:
    PathCPURenderEngine(Scene *scn, Film *flm, boost::mutex *filmMutex,
                        const Properties &cfg);
    virtual ~PathCPURenderEngine();
    
    
    void Start();
    
    void Interrupt();
    /*
    void Stop();
    */
    
    unsigned int GetPass() const;
    unsigned int GetThreadCount() const;
    RenderEngineType GetEngineType() const { return PATHCPU; }
    
    double GetTotalSamplesSec() const {
        return (m_elapsedTime == 0.0) ? 0.0 : (m_samplesCount / m_elapsedTime);
    }
    double GetRenderingTime() const { return (m_startTime == 0.0) ? 0.0 : (WallClockTime() - m_startTime); }

    void UpdateFilm();
    
    friend class PathCPURenderThread;
    friend class PathIntegrator;
    
    unsigned int m_samplePerPixel;
    
    // Signed because of the delta parameter
    int m_maxPathDepth;
    
    int m_rrDepth;
    float m_rrImportanceCap;
    
private:
    vector<PathCPURenderThread *> m_renderThreads;
    SampleBuffer *m_sampleBuffer;
    
    double m_screenRefreshInterval; // in seconds
    
    double m_startTime;
    double m_elapsedTime;
    unsigned long long m_samplesCount;
    
    bool m_enableOpenGLInterop, m_dynamicCamera;
};

#endif
