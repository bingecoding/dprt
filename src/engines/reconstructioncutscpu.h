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

#ifndef RECONSTRUCTIONCUTSCPU_H
#define RECONSTRUCTIONCUTSCPU_H

#include <vector>

#include "engines/renderengine.h"
#include "sampler.h"
#include "engines/vplgpu.h"
#include "lighttree.h"

#include "openclintersectiondevice.h"

#include <boost/thread/thread.hpp>

#define BLOCKSIZE 4 //4x4

using namespace std;

struct RayPixel {
    
    Ray pathRay;
    RayHit rayHit;
    Normal shadeN;
    bool intersect;
    int x;
    int y;
    
    TriangleMesh *mesh;
    unsigned int triIndex;
    Material *triMat;
};


class ReconstructionCutsCpuRenderEngine;

//------------------------------------------------------------------------------
// Lightcuts Integrator
//------------------------------------------------------------------------------

class ReconstructionCutsIntegrator {
public:
    ReconstructionCutsIntegrator(ReconstructionCutsCpuRenderEngine *renderEngine, Sampler *samp);
    ~ReconstructionCutsIntegrator();
    
    void Li();
    
    Spectrum GetMaterialTerm(Cluster *node, Point shadingPoint,
                             Normal shadingNormal,
                             const SurfaceMaterial *triSurfMat, const Vector wo);
    float GetVisibilityTerm(Cluster *node, Point shadingPoint);
    float GetGeometricTerm(Cluster *node, Point shadingPoint);
    
    float MaxCosine(Cluster *node, Point shadingPoint, Normal shadingNormal);
    
    float GetBoundGeometricTerm(Cluster *node, Point shadingPoint);
    
    void GetClusterRadiance(Cluster *node, Point shadingPoint,
                            Normal shadingNormal,
                            const SurfaceMaterial *triSurfMat, const Vector wo,
                            const int sample = -1);

    void GetPixelRadiance(const RayPixel &rayPixel, const int sample = -1);
    void GetReconstructionCut(const RayPixel &rayPixel,
                              const RayPixel *rayBlock,
                              const int pixelIdx);
    float GetReconstructionCutRadiance(Cluster *node,
                                      const RayPixel &rayPixel,
                                      const RayPixel *rayBlock,
                                       const int pixelIdx,
                                      const Point shadingPoint,
                                      const Normal shadingNormal,
                                      const SurfaceMaterial *triSurfMat,
                                      const Vector wo);
    Spectrum LightCutMode(Cluster *node,
                          const Point shadingPoint,
                          const Normal shadingNormal,
                          const SurfaceMaterial *triSurfMat,
                          const Vector wo);
    
    
    void SpectrumBilinearInterpolation(const Cluster *node,
                                       const RayPixel &rayPixel,
                                       const RayPixel *rayBlock,
                                       const int pixelIdx,
                                       Spectrum *interpolatedIntensity);
    void DirectionBilinearInterpolation(const Cluster *node,
                                        const RayPixel &rayPixel,
                                        const RayPixel *rayBlock,
                                        Vector *interpolatedDirection);
    
    double m_statsRenderingStart;
    double m_statsTotalSampleCount;

private:
    
    ReconstructionCutsCpuRenderEngine *m_renderEngine;
    Sampler *m_sampler;
    RandomGenerator *m_rnd;
    
    Scene *m_scene;
    //vector<VPL> m_virtualLights;
    
    SampleBuffer *m_sampleBuffer;
    
    LightTree *m_lightTree;
    unsigned int m_lightTreeSize;
    float m_lightPaths;
    
    struct CompareErrorBound {
        
        bool operator()(const Cluster *c1, const Cluster *c2) const {
            // The biggest error bound value will be at the front of the heap
            return c1->errorBound.Y() < c2->errorBound.Y();
        }
    };
    
    float ErrorRatio;
    
    float m_sceneDiag;
};


class ReconstructionCutsCpuRenderThread {
public:
    ReconstructionCutsCpuRenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                           const unsigned int samplePerPixel, ReconstructionCutsCpuRenderEngine *renderEngine);
    /*
    ~LightcutsCpuRenderThread();
    */
    void Start();
    void Interrupt();
    void Stop();
    
    //unsigned int GetPass() const { return sampler->GetPass(); }
    
    friend class ReconstructionCutsCpuRenderEngine;
    
protected:
    unsigned int m_threadIndex;
    ReconstructionCutsCpuRenderEngine *m_renderEngine;
    
private:
    static void RenderThreadImpl(ReconstructionCutsCpuRenderThread *renderThread);
    
    float m_samplingStart;
    Sampler *m_sampler;
    ReconstructionCutsIntegrator *m_reconstructionCutsIntegrator;
    //RayBuffer *rayBuffer;
    
    boost::thread *m_renderThread;
};

//------------------------------------------------------------------------------
// Path Tracing CPU-only render engine
//------------------------------------------------------------------------------

class ReconstructionCutsCpuRenderEngine : public RenderEngine {
public:
    ReconstructionCutsCpuRenderEngine(Scene *scn, Film *flm, boost::mutex *filmMutex,
                             const Properties &cfg);
    virtual ~ReconstructionCutsCpuRenderEngine();
    
    void Start();
    
    void Interrupt();
    
    void Stop();
    
    unsigned int GetPass() const;
    unsigned int GetThreadCount() const;
    RenderEngineType GetEngineType() const { return LIGHTCUTSCPU; }
    
    double GetTotalSamplesSec() const {
        return (m_elapsedTime == 0.0) ? 0.0 : (m_samplesCount / m_elapsedTime);
    }
    double GetRenderingTime() const { return (m_startTime == 0.0) ? 0.0 : (WallClockTime() - m_startTime); }

    void UpdateFilm();
    
    friend class ReconstructionCutsCpuRenderThread;
    friend class ReconstructionCutsCpuIntegrator;
    friend class ReconstructionCutsIntegrator;
    
    unsigned int m_samplePerPixel;
    
    // Signed because of the delta parameter
    int m_maxPathDepth;
    
    int m_rrDepth;
    float m_rrImportanceCap;
    float m_clamping;
    
private:
    vector<ReconstructionCutsCpuRenderThread *> m_renderThreads;
    SampleBuffer *m_sampleBuffer;
    
    double m_screenRefreshInterval; // in seconds
    
    double m_startTime;
    double m_elapsedTime;
    unsigned long long m_samplesCount;
    
    bool m_enableOpenGLInterop, m_dynamicCamera;
    
};

#endif
