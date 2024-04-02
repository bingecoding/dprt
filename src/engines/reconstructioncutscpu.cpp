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
#include <queue>

#include <boost/thread/mutex.hpp>

#include "engines/reconstructioncutscpu.h"
#include "raytracer.h"
#include "bvhaccel.h"
#include "buffer.h"
#include "sampler.h"

int hashk(uint32_t k) {
    
    float A = 0.5*(sqrt(5) - 1); // Suggested by Knuth
    uint32_t s = floor(A * 4294967296);
    uint32_t x = k*s;
    uint32_t p = 8; // Size of Hash table 2^p
    uint32_t sh = x >> (32-p);
    int r = sh;
    return r;
    
}

//------------------------------------------------------------------------------
// ReconstructionCutsCpuRenderEngine
//------------------------------------------------------------------------------

ReconstructionCutsCpuRenderEngine::ReconstructionCutsCpuRenderEngine(Scene *scene, Film *film,
                                                boost::mutex *filmMutex,
                                                const Properties &cfg) :
RenderEngine(scene, film, filmMutex, cfg)
{
    m_samplePerPixel = max(1, cfg.GetInt("path.sampler.spp"));
    m_samplePerPixel *=m_samplePerPixel;
    m_maxPathDepth = cfg.GetInt("path.maxdepth");
    m_rrDepth = cfg.GetInt("path.russianroulette.depth");
    m_rrImportanceCap = cfg.GetFloat("path.russianroulette.cap");
    
    m_clamping = cfg.GetFloat("vpl.clamping");
    if(m_clamping <= 0 || m_clamping > 1){
        m_clamping = 1.f;
    }
    
    m_startTime = 0.0;
    m_samplesCount = 0;
    
    m_sampleBuffer = m_film->GetFreeSampleBuffer();
    
    const unsigned int seedBase = (unsigned int)(WallClockTime() / 1000.0);
    
    // Create and start render threads
    const size_t renderThreadCount = 1;
    cerr << "Starting "<< renderThreadCount << " ReconstructionCutsCpu render threads"
    << endl;
    
    for (size_t i = 0; i < renderThreadCount; ++i) {
        ReconstructionCutsCpuRenderThread *t = new ReconstructionCutsCpuRenderThread(i,
                                                         seedBase,
                                                         i /(float)renderThreadCount,
                                                         m_samplePerPixel,
                                                         this);
        m_renderThreads.push_back(t);
    }
    
}

void ReconstructionCutsCpuRenderEngine::Start()
{
    RenderEngine::Start();
    
    m_samplesCount = 0;
    m_elapsedTime = 0.0f;
    
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Start();
    
    m_startTime = WallClockTime();
}

void ReconstructionCutsCpuRenderEngine::Stop()
{
    RenderEngine::Stop();

    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Stop();

    //UpdateFilm();
}

void ReconstructionCutsCpuRenderEngine::Interrupt()
{
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Interrupt();
}

ReconstructionCutsCpuRenderEngine::~ReconstructionCutsCpuRenderEngine()
{
}

unsigned int ReconstructionCutsCpuRenderEngine::GetPass() const {
    return m_samplesCount / (m_film->GetWidth() * m_film->GetHeight());
}

unsigned int ReconstructionCutsCpuRenderEngine::GetThreadCount() const {
    return m_renderThreads.size();
}

//------------------------------------------------------------------------------
// ReconstructionCutsCpuRenderThread
//------------------------------------------------------------------------------

ReconstructionCutsCpuRenderThread::ReconstructionCutsCpuRenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                                         const unsigned int samplePerPixel, ReconstructionCutsCpuRenderEngine *renderEngine)
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

    m_reconstructionCutsIntegrator = new ReconstructionCutsIntegrator(m_renderEngine, m_sampler);
    
}

void ReconstructionCutsCpuRenderThread::Start()
{

    
    const unsigned int startLine = Clamp<unsigned int>(
                                                       m_renderEngine->m_film->GetHeight() * m_samplingStart,
                                                       0, m_renderEngine->m_film->GetHeight() - 1);
     
    m_sampler->Init(m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(), startLine);
    
    // Create the thread for the rendering
    m_renderThread = new boost::thread(boost::bind(ReconstructionCutsCpuRenderThread::RenderThreadImpl, this));
}

void ReconstructionCutsCpuRenderThread::Stop() {
    
    if (m_renderThread) {
        m_renderThread->interrupt();
        m_renderThread->join();
        delete m_renderThread;
        m_renderThread = NULL;
    }
    
    //m_started = false;
    
    // frameBuffer is delete on the destructor to allow image saving after
    // the rendering is finished
}

void ReconstructionCutsCpuRenderThread::Interrupt() {
    if (m_renderThread)
        m_renderThread->interrupt();
}

void ReconstructionCutsCpuRenderThread::RenderThreadImpl(ReconstructionCutsCpuRenderThread *renderThread) {
    cerr << "[ReconstructionCutsCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread started" << endl;
    
    try {
        
        ReconstructionCutsIntegrator *ReconstructionCutsIntegrator = renderThread->m_reconstructionCutsIntegrator;
        //ReconstructionCutsIntegrator->Preprocess();
        while (!boost::this_thread::interruption_requested()) {
            ReconstructionCutsIntegrator->Li();
        }
        
        cerr << "[ReconstructionCutsCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    } catch (boost::thread_interrupted) {
        cerr << "[ReconstructionCutsCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    }

}

//------------------------------------------------------------------------------
// Path Integrator
//------------------------------------------------------------------------------
ReconstructionCutsIntegrator::ReconstructionCutsIntegrator(ReconstructionCutsCpuRenderEngine *re, Sampler *samp) :
m_renderEngine(re), m_sampler(samp)
{
    ErrorRatio = 0.02f;
    
    m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
    m_statsRenderingStart = WallClockTime();
    m_statsTotalSampleCount = 0;
    unsigned long seedBase = (unsigned long)(WallClockTime() / 1000.0);
    
    m_rnd = new RandomGenerator(seedBase);
    m_scene = re->m_scene;
    
    re->Preprocess();
    m_lightTree = re->m_lightTree;
    m_lightTreeSize = re->m_lightTree->m_lightTreeSize;
    m_lightPaths = re->m_lightPaths;
    
    BBox sceneBox = m_scene->m_bbox;
    Vector dv(sceneBox.pMax.x - sceneBox.pMin.x,
              sceneBox.pMax.y - sceneBox.pMin.y,
              sceneBox.pMax.z - sceneBox.pMin.z);
    
    m_sceneDiag = dv.Length();
    
}

void ReconstructionCutsIntegrator::
SpectrumBilinearInterpolation(const Cluster *node,
                              const RayPixel &rayPixel,
                              const RayPixel *rayBlock,
                              const int pixelIdx,
                              Spectrum *interpolatedIntensity)

{
    
    Spectrum intensity[SAMPLESIZE];
    
    for(int i=0; i < SAMPLESIZE; i++) {
        intensity[i] = node->sampleNode[i].intensity;
    }
    
    // Get x and y co-ordinates
    float x = pixelIdx % 4;
    float y = pixelIdx / 4;
    
    x += m_rnd->floatValue() -  0.5f;
    y += m_rnd->floatValue() -  0.5f;
    
    Spectrum p11 = intensity[0] * (3 - x) * (3 - y);
    Spectrum p12 = intensity[1] * (3 - x) * (y - 0);
    Spectrum p21 = intensity[2] * (x - 0) * (3 - y);
    Spectrum p22 = intensity[3] * (x - 0) * (y - 0);
    float weight = 1.f/((3 - 0) * (3 - 0));
    
    *interpolatedIntensity = weight * (p11 + p21 + p12 + p22);
    
}

float ReconstructionCutsIntegrator::GetBoundGeometricTerm(Cluster *node,
                            Point shadingPoint)
{
    float boundGeometricTerm;
    // Compute shortest distance to bounding box
    BBox bbox =  node->bounds;
    Point y = ClosestPointToBBox(shadingPoint, bbox);
    // Upper bound geometric term
    boundGeometricTerm = 1.f / DistanceSquared(y, shadingPoint);
    boundGeometricTerm = min(2.f/3.f, boundGeometricTerm);
    
    return boundGeometricTerm;
}

Spectrum ReconstructionCutsIntegrator::GetMaterialTerm(Cluster *node, Point shadingPoint,
                                        Normal shadingNormal,
                                        const SurfaceMaterial *triSurfMat, const Vector wo)
{
    Spectrum materialTerm(0.f);
  
    const Point vplPoint = node->representativeLight.hitPoint;
    const Vector wi(shadingPoint - vplPoint);

    // TODO -- When VPL is located on the light source, it should be ignored
    // At the moment we only have diffuse surfaces
    if (triSurfMat->IsDiffuse()) {
        materialTerm = triSurfMat->f(wo, wi, shadingNormal);// * surfaceColor;
    }
    
    return  materialTerm;
}

float ReconstructionCutsIntegrator::MaxCosine(Cluster *node, Point shadingPoint, Normal shadingNormal)
{
    
    // 1st translate points of bounding box to hitpoint; the hitpoint
    // is the new origin.
    BBox bbox =  node->bounds;
    Point bboxCorners[8];
    bboxCorners[0] = bbox.pMin;
    bboxCorners[1] = bbox.pMax;
    bboxCorners[2] = Point(bboxCorners[0].x, bboxCorners[0].y, bboxCorners[1].z);
    bboxCorners[3] = Point(bboxCorners[0].x, bboxCorners[1].y, bboxCorners[0].z);
    bboxCorners[4] = Point(bboxCorners[1].x, bboxCorners[0].y, bboxCorners[0].z);
    bboxCorners[5] = Point(bboxCorners[0].x, bboxCorners[1].y, bboxCorners[1].z);
    bboxCorners[6] = Point(bboxCorners[1].x, bboxCorners[0].y, bboxCorners[1].z);
    bboxCorners[7] = Point(bboxCorners[1].x, bboxCorners[1].y, bboxCorners[0].z);
    
    Transform translateBBox = Translate(Vector(-1.f * shadingPoint));
    for(int i=0;  i < 8; i++) {
        bboxCorners[i] = translateBBox(bboxCorners[i]);
    }
    
    Vector v = Vector(shadingNormal);
    Vector u = Normalize(v);
    Vector zAxis(0,0,1);
    // Find angle between shading normal and z-axis
    float alpha = acosf(Dot(u, zAxis)) * 180.0f / M_PI;
    
    BBox rotatedBBox;
    if(!(shadingNormal.x == 0.f && shadingNormal.y == 0.f)) {
        
        // Rotate points around shading normal which acts as the z-axis
        // The formula used is "rotation around an arbitrary axis" ;
        // see wikipedia http://en.wikipedia.org/wiki/Rotation_matrix
        // section Rotation matrix from axis and angle
        float temp = v.x;
        v.x = v.y;
        v.y = -1.f * temp;
        v.z = 0.f;
        
        Transform rotationMatrix = Rotate(alpha, v);
        for(int i=0;  i < 8; i++) {
            bboxCorners[i] = rotationMatrix(bboxCorners[i]);
        }
        
      
    }
    
    // Find new bounding points
    for(int i=0; i < 8; i++) {
        rotatedBBox = Union(rotatedBBox, bboxCorners[i]);
    }
    
    // Compute max cosine
    float maxCosine;
    float xMax = rotatedBBox.pMax.x;
    float yMax = rotatedBBox.pMax.y;
    float zMax = rotatedBBox.pMax.z;
    if( zMax >= 0.f ) {
        float xMin = rotatedBBox.pMin.x;
        float yMin = rotatedBBox.pMin.y;
        // See footnote 1 in original Lightcuts paper for the following.
        if( xMin*xMax < 0.f) {
            xMin = 0.f;
        }
        if( yMin*yMax < 0.f) {
            yMin = 0.f;
        }
        
        float d = sqrt(min(xMin*xMin, xMax*xMax) + min(yMin*yMin,yMax*yMax) + zMax*zMax);
        if(d == 0.f)
            maxCosine = 1.f;
        else
            maxCosine = zMax / d;
        
    }
    else {
        /*
        float xMin = rotatedBBox.pMin.x;
        float yMin = rotatedBBox.pMin.y;
        
        float d = sqrt(max(xMin*xMin, xMax*xMax) + max(yMin*yMin, yMax*yMax) + zMax*zMax);
        if( d == 0.f)
            maxCosine = 1.f;
        else
            maxCosine = zMax / d;
        */
        maxCosine = 0.f;
    }
    
    return maxCosine;

}

float ReconstructionCutsIntegrator::GetVisibilityTerm(Cluster *node, Point shadingPoint)
{
    
    float visibilityTerm = 0.f;
    
    Point vplPoint = node->representativeLight.hitPoint;
    Normal vplN = node->representativeLight.n;
    float d2 = DistanceSquared(shadingPoint, vplPoint);
    Vector wi = Normalize(vplPoint - shadingPoint);

    Ray vplShadowRay(shadingPoint, wi);
    vplShadowRay.maxt = sqrtf(d2) * (1.f - RAY_EPSILON);
    bool visible = !m_scene->IntersectP(&vplShadowRay);
    if (visible) {
        visibilityTerm = 1.0f;
    }

    return visibilityTerm;
}

float ReconstructionCutsIntegrator::GetGeometricTerm(Cluster *node, Point shadingPoint)
{

    Point vplPoint = node->representativeLight.hitPoint;
    float d2 = DistanceSquared(shadingPoint, vplPoint);
    float G = 1 / d2;
    G = min(2.f/3.f, G);
    
    return G;
    
}

void ReconstructionCutsIntegrator::GetClusterRadiance(Cluster *node, Point shadingPoint,
                                             Normal shadingNormal,
                                             const SurfaceMaterial *triSurfMat,
                                             const Vector wo, const int sample)
{
    
    float cosine, geometricTerm, visibilityTerm;
    Spectrum intensity, brdf, estimatedRadiance(0.f);
    
    intensity = node->intensity;
    visibilityTerm = GetVisibilityTerm(node, shadingPoint);
    geometricTerm = GetGeometricTerm(node, shadingPoint);
    brdf = GetMaterialTerm(node, shadingPoint, shadingNormal, triSurfMat, wo);
    
    // Bound terms
    float boundVisibilityTerm = 1.f;
    Spectrum boundMaterialTerm(0.f);
    float boundGeometricTerm;
    float maxCosine;
    maxCosine = 1.f;//MaxCosine(node, shadingPoint, shadingNormal);
    boundMaterialTerm = brdf * maxCosine;
    boundGeometricTerm = GetBoundGeometricTerm(node, shadingPoint);
    
    // Compute cosine for material term
    Vector vpl;
    Point vplPoint;
    vplPoint = node->representativeLight.hitPoint;
    Normal vplN = node->representativeLight.n;
    vpl = Normalize(Vector(vplPoint - shadingPoint));
    
    node->vplDir = vpl;
    
    cosine = max(0.f, Dot(Normalize(shadingNormal), vpl));
    //cosine = AbsDot(shadingNormal, vpl);
    //if(cosine >= 0.001f) {
        //cosine = max(0.f, (Dot(shadingNormal, vpl) * Dot(vplN, vpl * -1.f)));
    //}
    geometricTerm *= AbsDot(Normalize(vplN), vpl);
    //cosine = max(0.f, (Dot(shadingNormal, vpl) * Dot(vplN, vpl * -1.f)));
    estimatedRadiance = brdf * cosine * geometricTerm * visibilityTerm * intensity;
    
    node->geoVisInt = geometricTerm * visibilityTerm;
    
    node->brdf = brdf;// *cosine;
    node->estimatedRadiance = estimatedRadiance;
    node->errorBound = boundMaterialTerm * boundGeometricTerm * boundVisibilityTerm * intensity;
    
    node->repEstimatedRadiance = brdf * cosine * geometricTerm * visibilityTerm;
    node->repErrorBound = boundMaterialTerm * boundVisibilityTerm;
    
    // Reconstruction cut sample points
    if(sample != -1 && sample < SAMPLESIZE) {
        node->sampleNode[sample].sampleSet = true;
        node->sampleNode[sample].intensity = node->geoVisInt;
        node->sampleNode[sample].dir = vpl;
        node->sampleNode[sample].totalIntensity = 1.f / (brdf * cosine).Y();
    }
    
}

Spectrum ReconstructionCutsIntegrator::LightCutMode(Cluster *node,
                                                    const Point shadingPoint,
                                                    const Normal shadingNormal,
                                                    const SurfaceMaterial *triSurfMat,
                                                    const Vector wo)
{
    
    Spectrum totalRadiance(0.f);

    GetClusterRadiance(node, shadingPoint, shadingNormal, triSurfMat, wo);
    totalRadiance = node->estimatedRadiance;
    
    std::priority_queue<Cluster*, vector<Cluster*>, CompareErrorBound> priorityHeap;
    priorityHeap.push(node);
    int cutsize = 1;
    while(!priorityHeap.empty()) {
        Cluster* cluster = priorityHeap.top();
        
        if((cluster->errorBound.Y() <= (ErrorRatio * totalRadiance.Y())) || cutsize >= 1000)
        {
            break; // Found lightcut
        }
        
        priorityHeap.pop();
        totalRadiance -= cluster->estimatedRadiance;
        cutsize--;

        for(int i=0; i < ARITY; i++) {
            cutsize++;
            int idx = cluster->siblingIDs[i];
            if(idx == cluster->repId) {
                
                Cluster *sibling = cluster->siblings[i];
                
                sibling->estimatedRadiance = cluster->repEstimatedRadiance * sibling->intensity;
                sibling->repEstimatedRadiance = cluster->repEstimatedRadiance;
                sibling->errorBound = cluster->repErrorBound * GetBoundGeometricTerm(sibling, shadingPoint) * sibling->intensity;
                sibling->repErrorBound = cluster->repErrorBound;
                
                totalRadiance += sibling->estimatedRadiance;
                
                // Change upper error bound to
                // estimated radiance for individual light
                if(sibling->isLeaf) {
                    sibling->errorBound = sibling->estimatedRadiance;
                }
                //std::cout << "heap push: " << sibling->ID << std::endl;
                
                priorityHeap.push(sibling);
                
            } else if(cluster->siblings[i] != NULL) {
                
                Cluster *sibling = cluster->siblings[i];
                
                GetClusterRadiance(sibling, shadingPoint, shadingNormal,
                                   triSurfMat, wo);
                
               
                totalRadiance += sibling->estimatedRadiance;
                
                // Change upper error bound to
                // estimated radiance for individual light
                if(sibling->isLeaf) {
                    sibling->errorBound = sibling->estimatedRadiance;
                }
                //std::cout << "heap push: " << sibling->ID << std::endl;
                priorityHeap.push(sibling);
                
                
            }
        }
        
    }

    return totalRadiance;
    
}


void ReconstructionCutsIntegrator::GetReconstructionCut(const RayPixel &rayPixel,
                                                        const RayPixel *rayBlock,
                                                        const int pixelIdx)
{
    const float clamp = m_renderEngine->m_clamping;
    Cluster *root = m_lightTree->m_lightTree;
    
    Ray pathRay = rayPixel.pathRay;
    RayHit rayHit = rayPixel.rayHit;
    // Get the triangle
    const TriangleMesh *mesh = rayPixel.mesh;
    const unsigned int triIndex = rayPixel.triIndex;
    // Get the material
    const Material *triMat = rayPixel.triMat;
    
    // Get normal
    Normal shadeN = rayPixel.shadeN;
    const SurfaceMaterial *triSurfMat = (SurfaceMaterial *) triMat;
    const Point hitPoint = pathRay(rayHit.t);
    const Vector wo = -pathRay.d;
   
    bool skipVpls = false;
    Spectrum radiance(0.f, 0.f, 0.f);
    radiance = EstimateDirect(m_renderEngine->m_scene,
                              m_renderEngine->m_film,
                              pathRay, rayHit, triSurfMat,
                              hitPoint, shadeN, m_rnd, &skipVpls);
    
    if(skipVpls) {
        m_sampleBuffer->SplatSample(rayPixel.x, rayPixel.y, radiance);
        return;
    }
    
    Spectrum lightRadiance = radiance;
    Spectrum indirectRadiance(0.f);
    
    //std::cout << "radiance before: " << radiance.Filter() << " pixel" << i << std::endl;
    //------------------------------------------------------------------
    // Compute indirect illumination
    //------------------------------------------------------------------
    
    double startTime = WallClockTime();
    
    // Lightcuts implementation
    Spectrum totalRadiance(0.f);
    GetClusterRadiance(root, hitPoint, shadeN,
                       triSurfMat, wo);
    
    std::priority_queue<Cluster*, vector<Cluster*>, CompareErrorBound> priorityHeap;
    priorityHeap.push(root);
    int cutsize = 1;
    while(!priorityHeap.empty()) {
        
        Cluster* cluster = priorityHeap.top();
        
        //cutsize--;
        priorityHeap.pop();
        
        bool noSamples = true;
        for(int i=0; i < SAMPLESIZE; i++) {
            if(cluster->sampleNode[i].sampleSet)
                noSamples = false;
        }
        
        if(noSamples) {
            
            totalRadiance += LightCutMode(cluster, hitPoint, shadeN, triSurfMat, wo);
            
            /*
            GetClusterRadiance(cluster, hitPoint, shadeN, triSurfMat, wo);
            totalRadiance += cluster->estimatedRadiance;
            if(cluster->isLeaf) {
                cluster->errorBound = cluster->estimatedRadiance;
            }
            
            if((cluster->errorBound.Y() <= (ErrorRatio * totalRadiance.Y())) || cutsize >= 1000)
            {
                break;
            }

            for(int i=0; i < ARITY; i++) {
                if(cluster->siblings[i] != NULL) {
                    cutsize++;
                    
                    //GetClusterRadiance(cluster->siblings[i], hitPoint, shadeN, triSurfMat, wo);
                    //tr += cluster->siblings[i]->estimatedRadiance;
                    
                    // Change upper error bound to
                    // estimated radiance for individual light
                    //if(cluster->siblings[i]->isLeaf) {
                    //    cluster->siblings[i]->errorBound = cluster->siblings[i]->estimatedRadiance;
                    //}
                    
                    priorityHeap.push(cluster->siblings[i]);
                }
            }
            */
            
            continue;

        }

        Spectrum maxIntensity = 0.f;
        Spectrum minIntensity = std::numeric_limits<float>::infinity();
        float minTotalIntensity = std::numeric_limits<float>::infinity();
        Vector dir;
        float weight = 0.f;
        for(int i=0; i < SAMPLESIZE; i++) {
            if(maxIntensity.Y() < cluster->sampleNode[i].intensity.Y()) {
                maxIntensity = cluster->sampleNode[i].intensity;
            }
            
            if(minIntensity.Y() > cluster->sampleNode[i].intensity.Y()) {
                minIntensity = cluster->sampleNode[i].intensity;
            }
            
            if(minTotalIntensity > cluster->sampleNode[i].totalIntensity) {
                minTotalIntensity = cluster->sampleNode[i].totalIntensity;
            }
            dir += cluster->sampleNode[i].intensity.Y() * cluster->sampleNode[i].dir;
            weight += cluster->sampleNode[i].intensity.Y();
        }
        
        if(weight != 0.f)
            dir /= weight;
        dir = Normalize(dir);
        float threshold = ErrorRatio * minTotalIntensity;
        
        if(maxIntensity.Y() == 0.f) {
            cutsize++;
            //continue;
        } else if(((maxIntensity.Y() - minIntensity.Y() < threshold) && minIntensity.Y() > 0)) {
            cutsize++;
            
            Spectrum interpolatedIntensity;
            SpectrumBilinearInterpolation(cluster, rayPixel, rayBlock, pixelIdx,
                                          &interpolatedIntensity);
            
            // Compute cosine for material term
            float cosine = max(0.f, Dot(Normalize(shadeN), dir));
            Spectrum brdf = GetMaterialTerm(cluster, hitPoint, shadeN, triSurfMat, wo);
            
            cluster->estimatedRadiance = brdf * cosine * interpolatedIntensity;
            
            totalRadiance += cluster->estimatedRadiance;

            
        } else if((maxIntensity.Y() < threshold) || cluster->isLeaf) {
            cutsize++;
            // Refine - compute radiance in standard fashion
            GetClusterRadiance(cluster, hitPoint, shadeN, triSurfMat, wo);
            totalRadiance += cluster->estimatedRadiance;
            
        } else {
            // Refine
            for(int i=0; i < ARITY; i++) {
                if(cluster->siblings[i] != NULL) {
                    cutsize++;
                    priorityHeap.push(cluster->siblings[i]);
                }
            }
            
            
        }
        
        
    }

    //radiance += lightcutRadiance;
    totalRadiance /= m_lightPaths;
    radiance = totalRadiance + lightRadiance;
    
    m_sampleBuffer->SplatSample(rayPixel.x, rayPixel.y, radiance);
    
    //std::cout << "visible " << count << std::endl;
    
    // Check if the sample buffer is full
    if (m_sampleBuffer->IsFull()) {
        m_statsTotalSampleCount += m_sampleBuffer->GetSampleCount();
        
        // Splat all samples on the film
        m_renderEngine->m_film->SplatSampleBuffer(m_sampler->IsPreviewOver(),
                                                  m_sampleBuffer);
        m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
    }
}

void ReconstructionCutsIntegrator::GetPixelRadiance(const RayPixel &rayPixel,
                                                    const int sample)
{
    
    const float clamp = m_renderEngine->m_clamping;
    Cluster *root = m_lightTree->m_lightTree;
    
    Ray pathRay = rayPixel.pathRay;
    RayHit rayHit = rayPixel.rayHit;
    // Get the triangle
    const TriangleMesh *mesh = rayPixel.mesh;
    const unsigned int triIndex = rayPixel.triIndex;
    // Get the material
    const Material *triMat = rayPixel.triMat;
    
    // Get normal
    Normal shadeN = rayPixel.shadeN;
    const SurfaceMaterial *triSurfMat = (SurfaceMaterial *) triMat;
    const Point hitPoint = pathRay(rayHit.t);
    const Vector wo = -pathRay.d;
    
    bool skipVpls = false;
    Spectrum radiance(0.f, 0.f, 0.f);
    
    radiance = EstimateDirect(m_renderEngine->m_scene,
                              m_renderEngine->m_film,
                              pathRay, rayHit, triSurfMat,
                              hitPoint, shadeN, m_rnd, &skipVpls);
    
    if(skipVpls) {
        m_sampleBuffer->SplatSample(rayPixel.x, rayPixel.y, radiance);
        return;
    }
    
    Spectrum lightRadiance = radiance;
    Spectrum indirectRadiance(0.f);
    
    //std::cout << "radiance before: " << radiance.Filter() << " pixel" << i << std::endl;
    //------------------------------------------------------------------
    // Compute indirect illumination
    //------------------------------------------------------------------
    
    double startTime = WallClockTime();
    
    // Lightcuts implementation
    Spectrum estimatedRadiance(0.f), totalRadiance(0.f);
    
    GetClusterRadiance(root, hitPoint, shadeN, triSurfMat, wo);
    estimatedRadiance = root->estimatedRadiance;
    totalRadiance = estimatedRadiance;
    
    std::priority_queue<Cluster*, vector<Cluster*>, CompareErrorBound> priorityHeap;
    priorityHeap.push(root);
    int cutsize = 1;
    while(!priorityHeap.empty()) {
        
        Cluster* cluster = priorityHeap.top();
    
        if((cluster->errorBound.Y() <= (ErrorRatio * totalRadiance.Y())) || cutsize >= 1000)
        {
            break; // Found lightcut
        }
        
        priorityHeap.pop();
        totalRadiance -= cluster->estimatedRadiance;
        cutsize--;

        for(int i=0; i < ARITY; i++) {
            cutsize++;
            int idx = cluster->siblingIDs[i];
            if(idx == cluster->repId) {
                
                Cluster *sibling = cluster->siblings[i];
                
                sibling->estimatedRadiance = cluster->repEstimatedRadiance * sibling->intensity;
                sibling->repEstimatedRadiance = cluster->repEstimatedRadiance;
                sibling->errorBound = cluster->repErrorBound * GetBoundGeometricTerm(sibling, hitPoint) * sibling->intensity;
                sibling->repErrorBound = cluster->repErrorBound;
                sibling->geoVisInt = cluster->geoVisInt;
                sibling->brdf = cluster->brdf;
                
                totalRadiance += sibling->estimatedRadiance;
                
                // Change upper error bound to
                // estimated radiance for individual light
                if(sibling->isLeaf) {
                    sibling->errorBound = sibling->estimatedRadiance;
                }
                //std::cout << "heap push: " << sibling->ID << std::endl;
                
                priorityHeap.push(sibling);
                
            } else if(cluster->siblings[i] != NULL) {
                
                Cluster *sibling = cluster->siblings[i];
                
                GetClusterRadiance(sibling, hitPoint, shadeN,
                                   triSurfMat, wo, sample);
                
                estimatedRadiance = sibling->estimatedRadiance;
                totalRadiance += estimatedRadiance;
                
                // Change upper error bound to
                // estimated radiance for individual light
                if(sibling->isLeaf) {
                    sibling->errorBound = estimatedRadiance;
                }
                //std::cout << "heap push: " << sibling->ID << std::endl;
                priorityHeap.push(sibling);
                
                
            }
        }
        
    }

    // Compute only for sample pixels at corners of 4x4 block
    if(sample != -1) {
        int maxCutId=0;
        std::map<int, Cluster*> cut;
        while(!priorityHeap.empty()) {
            Cluster* cluster = priorityHeap.top();
            priorityHeap.pop();
            cut[cluster->ID] = cluster;
            if(maxCutId < cluster->ID)
                maxCutId = cluster->ID;
        }
        
        std::map<int, Cluster*>::iterator it;
        for(int i=0; i < m_lightTree->m_lightTreeSize; i++) {
            Cluster *node = m_lightTree->m_lightTreeFlatCluster[i];
            
            // Reset values for sample not on the cut
            it = cut.find(node->ID);
            if(it==cut.end()) {
                node->sampleNode[sample].intensity = 0.f;
                node->sampleNode[sample].totalIntensity = 0.f;
                node->sampleNode[sample].sampleSet = false;
                node->sampleNode[sample].dir = 0.f;
            }
        }
       
        std::map<int, Cluster*>::iterator itHeapMap;
        for(itHeapMap = cut.begin(); itHeapMap != cut.end(); itHeapMap++) {
            
            Cluster *cutNode = itHeapMap->second;
            int id = cutNode->ID;
            
            float parent = floor((id - 1.f) / 2.f);
            while(parent >= 0) {
                int tid = (int)parent;
                Cluster *node = m_lightTree->m_lightTreeFlatCluster[tid];
                
                //node->sampleNode[sample].sampleSet = true;
                node->sampleNode[sample].intensity += cutNode->estimatedRadiance;
                node->sampleNode[sample].dir += cutNode->estimatedRadiance.Y() * cutNode->vplDir;
                
                parent = floor((parent -1.f) / 2.f);
            }
            
        }
        
        for(itHeapMap = cut.begin(); itHeapMap != cut.end(); itHeapMap++) {
            
            Cluster *cutNode = itHeapMap->second;
            int id = cutNode->ID;
            
            cutNode->sampleNode[sample].totalIntensity *= root->sampleNode[sample].intensity.Y();
            
            float parent = floor((id - 1.f) / 2.f);
            while(parent >= 0) {
                int tid = (int)parent;
                Cluster *node = m_lightTree->m_lightTreeFlatCluster[tid];
                
                parent = floor((parent -1.f) / 2.f);
                if(node->sampleNode[sample].sampleSet == true) {
                    continue; // only need to set the node once
                }
                
                node->sampleNode[sample].sampleSet = true;
                
                if(node->sampleNode[sample].intensity.Y() != 0.f)
                    node->sampleNode[sample].dir /=  node->sampleNode[sample].intensity.Y();
                
                node->sampleNode[sample].dir = Normalize(node->sampleNode[sample].dir);
                
                float cosine = max(0.f, Dot(Normalize(shadeN), node->sampleNode[sample].dir));
                Spectrum material = node->brdf * cosine;
                if(material.r == 0.f)
                    material.r = 1.f;
                if(material.g == 0.f)
                    material.g = 1.f;
                if(material.b == 0.f)
                    material.b = 1.f;
                
                node->sampleNode[sample].intensity /= material;
                node->sampleNode[sample].totalIntensity = root->sampleNode[sample].intensity.Y() / material.Y();
                
            }
            
        }
        
    } // end if
    
    //radiance += lightcutRadiance;
    totalRadiance /= m_lightPaths;
    radiance = totalRadiance + lightRadiance;
    
    m_sampleBuffer->SplatSample(rayPixel.x, rayPixel.y, radiance);
    
    // Check if the sample buffer is full
    if (m_sampleBuffer->IsFull()) {
        m_statsTotalSampleCount += m_sampleBuffer->GetSampleCount();
        
        // Splat all samples on the film
        m_renderEngine->m_film->SplatSampleBuffer(m_sampler->IsPreviewOver(),
                                                  m_sampleBuffer);
        m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
    }
    
}

void ReconstructionCutsIntegrator::Li()
{

    Sample sample;
    m_sampler->GetNextSample(&sample);
    
    const unsigned int width = m_renderEngine->m_film->GetWidth();
    const unsigned int height = m_renderEngine->m_film->GetHeight();
    
    int wBlock = width / BLOCKSIZE;
    int hBlock = height / BLOCKSIZE;
   
    for(int i=0; i < hBlock; i++) {
        for(int j=0; j < wBlock; j++) {

            bool standardLightcuts = false;
            RayPixel *rayPixels =  new RayPixel[BLOCKSIZE*BLOCKSIZE];
            // Work on single 4x4 Block
            
            // Compute pixel coordinates in current block
            for(int k=0; k < BLOCKSIZE*BLOCKSIZE; k++) {
                // height in block
                int h = k / BLOCKSIZE;
                rayPixels[k].x = k % BLOCKSIZE + j*BLOCKSIZE;
                rayPixels[k].y = h + i*BLOCKSIZE;
                
            }
            
            // Determine hitpoints for block
            for(int rh=0; rh < BLOCKSIZE*BLOCKSIZE; rh++) {
                
                // Generate rays from camera
                const unsigned int x = rayPixels[rh].x;
                const unsigned int y = rayPixels[rh].y;
                const float scrX = x +  m_rnd->floatValue() -  0.5f;
                const float scrY = y +  m_rnd->floatValue() -  0.5f;
              
                //rayPixels[rh].x = scrX;
                //rayPixels[rh].y = scrY;
                
                m_renderEngine->m_scene->m_camera->GenerateRay(scrX, scrY, width, height,
                                                               &rayPixels[rh].pathRay, 0.f, 0.f, 0.f);
                
                if(m_scene->Intersect(&rayPixels[rh].pathRay, &rayPixels[rh].rayHit))
                {
                    
                    rayPixels[rh].intersect = true;
                    
                    const unsigned int currentTriangleIndex = rayPixels[rh].rayHit.index;
                    const unsigned int currentMeshIndex =  m_scene->GetAccelerator()->GetMeshID(currentTriangleIndex);
                    
                    // Get the triangle
                    rayPixels[rh].mesh = m_scene->m_objectMeshes[currentMeshIndex];
                    rayPixels[rh].triIndex = m_scene->GetAccelerator()->GetMeshTriangleID(currentTriangleIndex);
                    
                    // Get the material
                    rayPixels[rh].triMat = m_scene->m_objectMaterials[currentMeshIndex];
                    
                    assert(rayPixels[rh].mesh != NULL);
                    
                    // Determine shading Normal
                    const unsigned int triIndex = rayPixels[rh].triIndex;
                    const Ray pathRay = rayPixels[rh].pathRay;
                    const RayHit rayHit = rayPixels[rh].rayHit;
                    Normal N = rayPixels[rh].mesh->InterpolateTriNormal(triIndex, rayHit.b1, rayHit.b2);
                    Normal shadeN = (Dot(pathRay.d, N) > 0.f) ? -N : N;
                    rayPixels[rh].shadeN = shadeN;
                    
                }
                else
                {
                    // If eye ray hit nothing use standard lightcuts for this
                    // block
                    rayPixels[rh].intersect = false;
                    standardLightcuts = true;
                }
            }
            
            // Determine if we can use reconstruction cuts for current block
            // All rays must have normals that do not differ by more than 30°
            // All eye rays must have the same surface material
            // No point of a ray may lie in the cone of another ray point
            if(!standardLightcuts) {
                
                for(int th=0; th < BLOCKSIZE*BLOCKSIZE-1; th++) {
                    
                    // Compare adjacent rays with each other
                    
                    // Normal test
                    Normal n1 = Normalize(rayPixels[th].shadeN);
                    Normal n2 = Normalize(rayPixels[th+1].shadeN);
                    float angle = Degrees(acosf(Dot(n1, n2)));
                    
                    if(angle > 30) {
                        standardLightcuts = true;
                        break;
                    }
                    
                    // Surface material test
                    if(rayPixels[th].triMat->GetType() != rayPixels[th+1].triMat->GetType()) {
                        standardLightcuts = true;
                        break;
                    }
                    
                    // Cone test has to be done for all eye rays
                    const Point apex = rayPixels[th].pathRay(rayPixels[th].rayHit.t);
                    Normal dir = n1; // axis of cone
                   
                    float h = m_sceneDiag;
                    float r = h / tanf(Radians(60)); // radius of base of cone
                    for(int tc=0; tc < BLOCKSIZE*BLOCKSIZE; tc++) {
                        if(tc == th) continue;
                
                        Point p = rayPixels[tc].pathRay(rayPixels[tc].rayHit.t);
                        float dist = Dot(p - apex, dir);
                        
                        if(0 <= dist <= h){
                            float radius = (dist / h) * r;
                            Vector v = Vector(p - apex) - Vector(dist * dir);
                            float orth= v.Length();
                            if(orth < radius) {
                                // Point inside cone
                                standardLightcuts = true;
                            }
                        }else {
                            standardLightcuts = true;
                        }
                    }
                    
                    if(standardLightcuts)
                        break;
                }
            }
        
            if(!standardLightcuts) {
                // Determine radiance for sample pixels in corner
                for(int sp=0; sp < SAMPLESIZE; sp++) {
                    
                    int sampleIndex = (sp == 0) ? 0 : (sp == 1) ? 3 : (sp==2) ? 12 : 15;
                    if(!rayPixels[sampleIndex].intersect) // Don't need to test for this
                        continue;
                    
                    GetPixelRadiance(rayPixels[sampleIndex], sp);
                    
                }
                // Determine remaining pixels using reconstruction cuts
                for(int np=0; np < BLOCKSIZE*BLOCKSIZE; np++) {
                    if(!rayPixels[np].intersect) // Don't need to test for this
                        continue;
                    bool skip = (np == 0) ? true : (np == 3) ? true : (np==12) ? true : (np==15) ? true : false;
                    if(!skip) {
                        GetReconstructionCut(rayPixels[np], rayPixels, np);
                        //GetPixelRadiance(rayPixels[np]);
                    }
                }
            }
            
            // Normal lightcuts
            if(standardLightcuts) {
                for(int p=0; p < BLOCKSIZE*BLOCKSIZE; p++) {
                    if(!rayPixels[p].intersect)
                        continue;
                    GetPixelRadiance(rayPixels[p]);
                }
            }
    
            delete [] rayPixels;
            
        } // end wBlock

    } // end hBlock



    
}

