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

#include "engines/vplcpu.h"
#include "raytracer.h"
#include "bvhaccel.h"
#include "buffer.h"
#include "sampler.h"

static float maxIntensity = 0.f;

Spectrum GetColour(double v,double vmin,double vmax)
{
    Spectrum c(1.0,1.0,1.0);//white
    double dv;
    
    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;
    
    if (v < (vmin + 0.25 * dv)) {
        c.r = 0;
        c.g = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c.r = 0;
        c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c.r = 4 * (v - vmin - 0.5 * dv) / dv;
        c.b = 0;
    } else {
        c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c.b = 0;
    }
    
    return(c);
}

//------------------------------------------------------------------------------
// VplCpuRenderEngine
//------------------------------------------------------------------------------

VplCpuRenderEngine::VplCpuRenderEngine(Scene *scene, Film *film,
                                       boost::mutex *filmMutex,
                                       const Properties &cfg,
                                       bool lightCuts) :
RenderEngine(scene, film, filmMutex, cfg)
{
    m_samplePerPixel = max(1, cfg.GetInt("path.sampler.spp"));
    m_samplePerPixel *=m_samplePerPixel;
    m_maxPathDepth = cfg.GetInt("path.maxdepth");
    m_rrDepth = cfg.GetInt("path.russianroulette.depth");
    m_rrImportanceCap = cfg.GetFloat("path.russianroulette.cap");
    
    m_lightTreeCutSize = cfg.GetInt("light.tree.cut.size");
    
    m_clamping = cfg.GetFloat("vpl.clamping");
    if(m_clamping <= 0 || m_clamping > 1){
        m_clamping = 1.f;
    }
    
    m_startTime = 0.0;
    m_samplesCount = 0;
    
    m_sampleBuffer = m_film->GetFreeSampleBuffer();
    
    m_lightCutsMode = lightCuts;
    
    const unsigned int seedBase = (unsigned int)(WallClockTime() / 1000.0);
    
    // Create and start render threads
    const size_t renderThreadCount = 1;
    cerr << "Starting "<< renderThreadCount << " VplCpu render threads"
    << endl;
    
    for (size_t i = 0; i < renderThreadCount; ++i) {
        VplCpuRenderThread *t = new VplCpuRenderThread(i,
                                                         seedBase,
                                                         i /(float)renderThreadCount,
                                                         m_samplePerPixel,
                                                         this);
        m_renderThreads.push_back(t);
    }
    
}

void VplCpuRenderEngine::Start()
{
    RenderEngine::Start();
    
    m_samplesCount = 0;
    m_elapsedTime = 0.0f;
    
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Start();
    
    m_startTime = WallClockTime();
}

void VplCpuRenderEngine::Stop()
{
    RenderEngine::Stop();

    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Stop();

    //UpdateFilm();
}

void VplCpuRenderEngine::Interrupt()
{
    for (size_t i = 0; i < m_renderThreads.size(); ++i)
        m_renderThreads[i]->Interrupt();
}

VplCpuRenderEngine::~VplCpuRenderEngine()
{
}

unsigned int VplCpuRenderEngine::GetPass() const {
    return m_samplesCount / (m_film->GetWidth() * m_film->GetHeight());
}

unsigned int VplCpuRenderEngine::GetThreadCount() const {
    return m_renderThreads.size();
}

//------------------------------------------------------------------------------
// VplCpuRenderThread
//------------------------------------------------------------------------------

VplCpuRenderThread::VplCpuRenderThread(unsigned int index, unsigned long seedBase, const float samplingStart,
                                       const unsigned int samplePerPixel, VplCpuRenderEngine *renderEngine)
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

    m_vplIntegrator = new VplIntegrator(m_renderEngine, m_sampler);
    
}

void VplCpuRenderThread::Start()
{

    
    const unsigned int startLine = Clamp<unsigned int>(
                                                       m_renderEngine->m_film->GetHeight() * m_samplingStart,
                                                       0, m_renderEngine->m_film->GetHeight() - 1);
     
    m_sampler->Init(m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(), startLine);
    
    // Create the thread for the rendering
    m_renderThread = new boost::thread(boost::bind(VplCpuRenderThread::RenderThreadImpl, this));
}

void VplCpuRenderThread::Stop() {
    
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

void VplCpuRenderThread::Interrupt() {
    if (m_renderThread)
        m_renderThread->interrupt();
}

void VplCpuRenderThread::RenderThreadImpl(VplCpuRenderThread *renderThread) {
    cerr << "[VplCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread started" << endl;
    
    try {
        renderThread->m_renderEngine->Preprocess();
        VplIntegrator *vplIntegrator = renderThread->m_vplIntegrator;
        //vplIntegrator->Preprocess();
        while (!boost::this_thread::interruption_requested()) {
            vplIntegrator->Li();
        }
        
        cerr << "[VplCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    } catch (boost::thread_interrupted) {
        cerr << "[VplCpuRenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
    }

}

//------------------------------------------------------------------------------
// Path Integrator
//------------------------------------------------------------------------------
VplIntegrator::VplIntegrator(VplCpuRenderEngine *re, Sampler *samp) :
m_renderEngine(re), m_sampler(samp)
{
    m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
    m_statsRenderingStart = WallClockTime();
    m_statsTotalSampleCount = 0;
    unsigned long seedBase = (unsigned long)(WallClockTime() / 1000.0);
    
    m_rnd = new RandomGenerator(seedBase);
    m_scene = re->m_scene;
}

float VplIntegrator::MaxCosine(Cluster *node, Point shadingPoint,
                               Normal shadingNormal)
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
    
    // Find angle between shading normal and z-axis
    Vector v = Vector(shadingNormal);
    Vector u = Normalize(v);
    Vector zAxis(0,0,1);
    float alpha = acosf(Dot(u, zAxis)) * 180.0f / M_PI;
    
    // Project shading normal onto XY plane
    Vector psxy;
    psxy.x = shadingNormal.x;
    psxy.y = shadingNormal.y;
    psxy.z = 0.f;
    
    Vector nua = Cross(psxy, zAxis);
    Transform rotation = Rotate(alpha, nua);
    
    for(int i=0;  i < 8; i++) {
        bboxCorners[i] = rotation(bboxCorners[i]);
    }
    
     
    BBox rotatedBBox;
    for(int i=0; i < 8; i++) {
        rotatedBBox = Union(rotatedBBox, bboxCorners[i]);
    }
    
    // Compute max cosine
    float maxCosine = 0.f;
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
            maxCosine = 0.f;
        else
            maxCosine = zMax / d;
     
    }
    
    return maxCosine;
    
}

Spectrum VplIntegrator::GetMaterialTerm(Cluster *node, Point shadingPoint,
                                        Normal shadingNormal,
                                        const SurfaceMaterial *triSurfMat, const Vector wo)
{
    Spectrum materialTerm(0.f);

    const Point vplPoint = node->representativeLight.hitPoint;
    const Vector wi(shadingPoint - vplPoint);
    materialTerm = triSurfMat->f(wo, wi, shadingNormal);// * surfaceColor;

    return  materialTerm;
}

Spectrum VplIntegrator::GetBoundMaterialTerm(Cluster *node, Point shadingPoint,
                              Normal shadingNormal,
                              const SurfaceMaterial *triSurfMat, const Vector wo)
{
    Spectrum materialTerm(0.f);
    float maxCos = 1.f;//MaxCosine(node, shadingPoint, shadingNormal);
    
    if(triSurfMat->GetType() == MATTE) {
        
        MatteMaterial *mat = (MatteMaterial *)triSurfMat;
        Spectrum diffuse = mat->GetKdOverPI();
        
        materialTerm = diffuse * maxCos;
        
    } else if(triSurfMat->GetType() == BLINNPHONG) {
        
        BlinnPhongMaterial *mat = (BlinnPhongMaterial *)triSurfMat;
        Spectrum spec = mat->GetKs();
        Spectrum diff = mat->GetKd();
        
        float maxAlpha = 1.f;//MaxCosine(node, shadingPoint, Normal(wo));
        
        //float costheta = halfAngle;
        //float D = (mat->GetExp()+2) * INV_TWOPI * powf(costheta, mat->GetExp());
        //spec *= D;
        
        float specAngle = maxAlpha;
        float specExp = powf(specAngle, mat->GetExp());
        spec *= specExp;
        diff *= maxCos;
        
        materialTerm = diff + spec;
        
    }
    
    return materialTerm;
}

float VplIntegrator::GetVisibilityTerm(Cluster *node, Point shadingPoint)
{
    
    float visibilityTerm = 0.f;
    
    const Point vplPoint = node->representativeLight.hitPoint;
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

float VplIntegrator::GetBoundGeometricTerm(Cluster *node,
                                           Point &shadingPoint,
                                           Normal &shadingNormal)
{
    float boundGeometricTerm;
    // Compute shortest distance to bounding box
    BBox bbox =  node->bounds;
    Point y = ClosestPointToBBox(shadingPoint, bbox);
    // Upper bound geometric term
    boundGeometricTerm = 1.f / DistanceSquared(y, shadingPoint);
    boundGeometricTerm = min(.66f, boundGeometricTerm);
    return boundGeometricTerm;
}

float VplIntegrator::GetGeometricTerm(Cluster *node, Point shadingPoint)
{
    const Point vplPoint = node->representativeLight.hitPoint;
    float d2 = DistanceSquared(shadingPoint, vplPoint);
    float G = 1 / d2;
    G = min(.66f, G);
    return G;
}

void VplIntegrator::GetClusterRadiance(Cluster *node, Point shadingPoint,
                                       Normal shadingNormal,
                                       const SurfaceMaterial *triSurfMat,
                                       const Vector wo)
{
    
    float cosine, geometricTerm, visibilityTerm;
    Spectrum intensity, materialTerm, estimatedRadiance(0.f);
    
    intensity = node->intensity;
    materialTerm = GetMaterialTerm(node, shadingPoint, shadingNormal,
                                   triSurfMat, wo);
    
    // Bound terms
    float boundVisibilityTerm = 1.f;
    Spectrum boundMaterialTerm(0.f);
    float boundGeometricTerm;
    boundGeometricTerm = GetBoundGeometricTerm(node, shadingPoint, shadingNormal);
    boundMaterialTerm = GetBoundMaterialTerm(node, shadingPoint, shadingNormal,
                                            triSurfMat, wo);
    
    //geometricTerm = GetGeometricTerm(node, shadingPoint);
   
    // Compute cosine for material term
    Vector vpl;
    Point vplPoint;
    vplPoint = node->representativeLight.hitPoint;
    Normal vplN = node->representativeLight.n;
    vpl = Normalize(Vector(vplPoint - shadingPoint));
    
    float d2 = DistanceSquared(shadingPoint, vplPoint);
    cosine = max(0.f, Dot(Normalize(shadingNormal), vpl) * AbsDot(vpl, vplN));
    geometricTerm = cosine / d2;
    geometricTerm = min(.66f, geometricTerm);
    
    estimatedRadiance = materialTerm * geometricTerm * intensity;
    if(estimatedRadiance.Y() != 0.f || !estimatedRadiance.Black()) {
        visibilityTerm = GetVisibilityTerm(node, shadingPoint);
        estimatedRadiance *= visibilityTerm;
    }
    
    node->estimatedRadiance = estimatedRadiance;
    node->errorBound = boundMaterialTerm * boundGeometricTerm * boundVisibilityTerm * intensity;
}

void VplIntegrator::Li()
{
    double bvhTotalTime = 0.f;
    
    Sample sample;
    m_sampler->GetNextSample(&sample);
    
    const unsigned int width = m_renderEngine->m_film->GetWidth();
    const unsigned int height = m_renderEngine->m_film->GetHeight();
    
    const unsigned int pixelCount = width * height;
    const float clamp = m_renderEngine->m_clamping;
    
    Cluster *lightTree = m_renderEngine->m_lightTree->m_lightTree;
    const unsigned int lightTreeSize = m_renderEngine->m_lightTree->m_lightTreeSize;
    const std::vector<VPL> virtualLights = m_renderEngine->m_virtualLights;
    const float lightPaths = m_renderEngine->m_lightPaths;
    unsigned int cuts = 0;
    
    for (unsigned int i = 0; i < pixelCount; ++i) {
    
        const unsigned int width = m_renderEngine->m_film->GetWidth();
        const unsigned int height = m_renderEngine->m_film->GetHeight();
        
        Ray pathRay;
        RayHit rayHit;
        /*
         m_renderEngine->m_scene->m_camera->GenerateRay(
         sample.screenX, sample.screenY,
         m_renderEngine->m_film->GetWidth(), m_renderEngine->m_film->GetHeight(), &pathRay,
         m_sampler->GetLazyValue(&sample), m_sampler->GetLazyValue(&sample), m_sampler->GetLazyValue(&sample));
         */
        // Generate rays from camera
        const unsigned int x = i % width;
        const unsigned int y = i / width;
        const float scrX = x + m_rnd->floatValue() - 0.5f;
        const float scrY = y + m_rnd->floatValue() - 0.5f;
        m_scene->m_camera->GenerateRay(scrX, scrY, width, height, &pathRay,
                                       m_rnd->floatValue(), m_rnd->floatValue(),
                                       m_rnd->floatValue());
        
        Spectrum radiance(0.f, 0.f, 0.f);
        if(m_scene->Intersect(&pathRay, &rayHit)) {
            
            const unsigned int currentTriangleIndex = rayHit.index;
            const unsigned int currentMeshIndex = m_scene->GetAccelerator()->GetMeshID(currentTriangleIndex);
            
            // Get the triangle
            const TriangleMesh *mesh = m_scene->m_objectMeshes[currentMeshIndex];
            const unsigned int triIndex = m_scene->GetAccelerator()->GetMeshTriangleID(currentTriangleIndex);
            
            // Get the material
            const Material *triMat = m_scene->m_objectMaterials[currentMeshIndex];
            const SurfaceMaterial *triSurfMat = (SurfaceMaterial *) triMat;
            
            const Point hitPoint = pathRay(rayHit.t);
            const Vector wo = -pathRay.d;
            Normal N = mesh->InterpolateTriNormal(triIndex, rayHit.b1, rayHit.b2);
            Normal shadeN = (Dot(pathRay.d, N) > 0.f) ? -N : N;
            
            bool skipVpls = false;
            radiance = EstimateDirect(m_renderEngine->m_scene,
                                      m_renderEngine->m_film,
                                      pathRay, rayHit, triSurfMat,
                                      hitPoint, shadeN, m_rnd, &skipVpls);
            
            if(skipVpls) {
                m_sampleBuffer->SplatSample(scrX, scrY, radiance);
                continue;
            }
            
            Spectrum directLight = radiance;
            
            //std::cout << "radiance before: " << radiance.Filter() << " pixel" << i << std::endl;
            //------------------------------------------------------------------
            // Compute indirect illumination
            //------------------------------------------------------------------
            
            double startTime = WallClockTime();
            
            if(!m_renderEngine->m_lightCutsMode) {
                
                for (uint32_t j = 0; j < virtualLights.size(); ++j) {
                    
                    const VPL vpl = virtualLights[j];
                    float d2 = DistanceSquared(hitPoint, vpl.hitPoint);
                    Vector wi = Normalize(vpl.hitPoint - hitPoint);
                    float G = max(0.f,Dot(wi, shadeN)) * AbsDot(wi, vpl.n) / d2;//AbsDot(wi, shadeN) * AbsDot(wi, vpl.n) / d2;
                    //float G = AbsDot(wi, shadeN) * AbsDot(wi, vpl.n) / d2;
                    G = min(G, .66f);
                    Spectrum f = triSurfMat->f(wo, wi, shadeN);// * surfaceColor;
                    Spectrum Llight = f * G * vpl.contrib / (lightPaths);
                    if (G == 0.f || f.Black()) continue;
                    
                    if(Llight.Y() < 0.0001f) {
                        float continueProbability = .1f;
                        if(m_rnd->floatValue() > continueProbability) {
                            continue;
                        }
                        Llight /= continueProbability;
                    }
                    
                    Ray vplShadowRay(hitPoint, wi);
                    vplShadowRay.maxt = sqrtf(d2) * (1.f - RAY_EPSILON);
                    bool visible = !m_scene->IntersectP(&vplShadowRay);
                    if (visible) {
                        radiance += Llight;
                    }
                
                }
                
            } else {
                
                // Lightcuts implementation
                float k = 0.02f; // Threshold error 0.02
                Spectrum estimatedRadiance(0.f), totalRadiance(0.f), er(0.f), tr(0.f);
                
                //LightCluster **lightTree = m_lightTree->m_lightTreeArray; // 1st element is the root
                GetClusterRadiance(lightTree, hitPoint, shadeN, triSurfMat, wo);
                estimatedRadiance = lightTree->estimatedRadiance;
                totalRadiance = estimatedRadiance;
                
                //GetClusterRadiance(rootTree, hitPoint, shadeN, triSurfMat, wo);
                //estimatedRadiance = rootTree->estimatedRadiance;
                //totalRadiance = estimatedRadiance;
                
                //float weight = 1.f / rootTree->intensity.Y();
                std::priority_queue<Cluster*, vector<Cluster*>, CompareErrorBound> heap;
                heap.push(lightTree);
                
                //PriorityQueue spends most of its time allocating memory, need to optimise this!!!
                //PriorityQueue<LightCluster *, ClusterErrorBound> priorityQueue(lightTreeSize);
                //priorityQueue.Push(lightTree[0]);
                size_t cutsize = 1;
                std::vector<Spectrum> iLights;
                //while(!priorityQueue.Empty()) {
                while(!heap.empty()) {
                    
                    //LightCluster *cluster = priorityQueue.Top();
                    Cluster *cluster = heap.top();
                    
                    float re = cluster->errorBound.Y();// / abs(cluster->estimatedRadiance.Y() - 0.5f);
                    if( re <= k * totalRadiance.Y() || cutsize >= m_renderEngine->m_lightTreeCutSize)
                    {
                        break; // Found lightcut
                    }
                    
                    heap.pop();
                    //priorityQueue.Pop();
                    //cutsize--;
                    totalRadiance -= cluster->estimatedRadiance;
                    totalRadiance.Clamp(0.f);
                    
                    for(int i=0; i < ARITY; i++) {
                        
                        int idx = cluster->siblingIDs[i];
                        if(idx != -1) {
                            
                            Cluster *sibling = cluster->siblings[i];
                            
                            GetClusterRadiance(sibling, hitPoint, shadeN,
                                               triSurfMat, wo);
                            
                            estimatedRadiance = sibling->estimatedRadiance;
                            
                            // Change upper error bound to
                            // estimated radiance for individual light
                            if(sibling->isLeaf) {
                                sibling->errorBound = estimatedRadiance;
                            }
                            
                            totalRadiance += estimatedRadiance;
                            if(sibling->estimatedRadiance.Y() != 0.f) {
                                cutsize++;
                                cuts++;
                            }
                            
                            //priorityQueue.Push(sibling);
                            heap.push(sibling);
                        }
                    }
                    
                }
                totalRadiance /= lightPaths;
                radiance = totalRadiance + directLight;
                
                
                //float af = 1.0 / (float)m_renderEngine->m_lightTreeCutSize;
                //float f = af * cutsize;
                //radiance = GetColour(f, 0.f, 1.f);
                
            }
            
            m_sampleBuffer->SplatSample(scrX, scrY, radiance);
        
        }

        // Check if the sample buffer is full
        if (m_sampleBuffer->IsFull()) {
            m_statsTotalSampleCount += m_sampleBuffer->GetSampleCount();
            
            // Splat all samples on the film
            m_renderEngine->m_film->SplatSampleBuffer(m_sampler->IsPreviewOver(), m_sampleBuffer);
            m_sampleBuffer = m_renderEngine->m_film->GetFreeSampleBuffer();
        }
        
    }
    
    std::cout << "Cuts " << cuts << std::endl;
    
}

