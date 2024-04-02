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

#include "renderengine.h"

void RenderEngine::Preprocess()
{
    RenderEngineType ret = GetEngineType();
    float multiplier = 3.25f;
    if(ret == VPLGPU || ret == VPLCPU)
        multiplier = 2.f;
    if(ret == RECONSTRUCTIONCUTSGPU || ret == RECONSTRUCTIONCUTSCPU)
        multiplier = 4.75f;
    
    for(int i=0; i < m_lightPaths/2; i++) {
        // Select the light to sample
        float lightStrategyPdf;
        const LightSource *light = m_scene->SampleAllLights(m_rnd->floatValue(),
                                                            &lightStrategyPdf);
        
        Ray ray;
        float lightPdf;
        // Sample ray leaving light source
        Spectrum alpha = light->Sample_L(m_scene, m_rnd->floatValue(), m_rnd->floatValue(),
                                         m_rnd->floatValue(), m_rnd->floatValue(),
                                         m_rnd->floatValue(), &lightPdf, &ray);
        
        if(lightPdf == 0.f || alpha.Black()) {
            continue;
        }
        
        alpha /=  lightPdf * lightStrategyPdf;
        //alpha *= 2.f;
        RayHit rayHit;
        bool firstVpl = true;
        for (int j=0; j < m_depth; j++) {
            
            if(m_scene->Intersect(&ray, &rayHit)) {
                
                const unsigned int currentTriangleIndex = rayHit.index;
                const unsigned int currentMeshIndex = m_scene->GetAccelerator()->GetMeshID(currentTriangleIndex);
                
                // Get the triangle
                const TriangleMesh *mesh = m_scene->m_objectMeshes[currentMeshIndex];
                const unsigned int triIndex = m_scene->GetAccelerator()->GetMeshTriangleID(currentTriangleIndex);
                
                // Get the material
                const Material *triMat = m_scene->m_objectMaterials[currentMeshIndex];
                
                // We don't want a vpl to be at a light source
                if((triMat->IsLightSource())) {
                    break;
                }
                
                const SurfaceMaterial *triSurfMat = (SurfaceMaterial *) triMat;
                const Point hitPoint = ray(rayHit.t);
                Vector wo = -ray.d;
                
                Spectrum surfaceColor;
                if (mesh->HasColors())
                    surfaceColor = mesh->InterpolateTriColor(triIndex, rayHit.b1, rayHit.b2);
                else
                    surfaceColor = Spectrum(1.f, 1.f, 1.f);
                
                // Interpolate face normal
                Normal N = mesh->InterpolateTriNormal(triIndex, rayHit.b1, rayHit.b2);
                // Flip the normal if required
                Normal shadeN = (Dot(ray.d, N) > 0.f) ? -N : N;
                
                // We assume the VPL is on a lambertian surface
                Spectrum contribution;
                if(firstVpl) {
                    // was 2.xxf instead of x.xxf
                    contribution = alpha * multiplier * triSurfMat->rho(wo, N, shadeN, m_rnd) * AbsDot(wo, shadeN) / M_PI;
                    //contribution = alpha * 7.25f * triSurfMat->rho(wo, N, shadeN, m_rnd) * AbsDot(wo, shadeN) / M_PI;
                    //contribution = alpha * triSurfMat->rho(wo, N, shadeN, m_rnd) * AbsDot(wo, shadeN) / M_PI;
                } else {
                    contribution = alpha * multiplier * triSurfMat->rho(wo, N, shadeN, m_rnd) / M_PI;
                    //contribution = alpha * 7.15f * triSurfMat->rho(wo, N, shadeN, m_rnd) / M_PI;
                    //contribution = alpha * triSurfMat->rho(wo, N, shadeN, m_rnd) / M_PI;
                }
                
                VPL vpl;
                vpl.contrib = contribution;
                vpl.hitPoint = hitPoint;
                vpl.pathHit = rayHit;
                vpl.n = shadeN;
                vpl.depth = j;
                
                //std::cout << "vpl contribution " << contribution.Filter() << " depth " << j << std::endl;
                
                m_virtualLights.push_back(vpl);
                
                // Sample new ray direction and update weight for virtual light path
                Vector wi;
                float pdf;
                bool specularBounce = false;
                const Spectrum fr = triSurfMat->Sample_f(wo, &wi, N, shadeN,
                                                         m_rnd->floatValue(), m_rnd->floatValue(), m_rnd->floatValue(),
                                                         false, &pdf, specularBounce) * surfaceColor;
                
                if (fr.Black() || pdf <= 0.f) {
                    break;
                }
                
                Spectrum contribScale = fr * AbsDot(wo, shadeN) / pdf;
                
                // Possibly terminate virtual light path with Russian roulette
                float rrProb = min(1.f, contribScale.Y());
                if (m_rnd->floatValue() > rrProb) {
                    break;
                }
                
                if(firstVpl) {
                    firstVpl = false;
                    alpha *=  fr * AbsDot(wo, shadeN) / rrProb;
                }else{
                    alpha *= contribScale / rrProb;
                }
                
                // New ray direction
                ray.o = hitPoint;
                ray.d = wi;
            }
            
        }
        
    }
    
    if(m_virtualLights.empty()) {
        throw runtime_error("No VPLs: set light.paths and light.depth; does the light source material have any ambience?");
    }
    
    double startTime = WallClockTime();
    m_lightTree = new LightTree(m_virtualLights);
    m_lightTree->BuildLightTree();
    double elapsedTime = WallClockTime() - startTime;
    cerr << "Number of VPLs: " << m_virtualLights.size() << endl;
    cerr << "Global light tree size: " << m_lightTree->m_lightTreeSize << " build time: " << int(elapsedTime * 1000.0) << " ms" << endl;
    
}

Spectrum EstimateDirect(Scene *scene, const Film *film, const Ray &pathRay,
                        const RayHit &rayHit, const SurfaceMaterial *triSurfMat,
                        const Point &hitpoint, const Normal &shadeN,
                        RandomGenerator *rnd, bool *skipVpls)
{
    
    Spectrum radiance(0.f, 0.f, 0.f);
        
    const unsigned int currentTriangleIndex = rayHit.index;
    const unsigned int currentMeshIndex =  scene->GetAccelerator()->GetMeshID(currentTriangleIndex);
    
    // Get the triangle
    const TriangleMesh *mesh = scene->m_objectMeshes[currentMeshIndex];
    const unsigned int triIndex = scene->GetAccelerator()->GetMeshTriangleID(currentTriangleIndex);
    
    // Get the material
    const Material *triMat = scene->m_objectMaterials[currentMeshIndex];
    
    // Compute emitted light if ray hit an area light source
    if((triMat->IsLightSource())) {
        const LightMaterial *mLight = (LightMaterial *)triMat;
        Spectrum Le = mLight->Le(mesh, triIndex, -pathRay.d);
        radiance += Le * 1.f;
        *skipVpls = true;
        return radiance;
    }
    
    const Vector wo = -pathRay.d;
    
    Spectrum surfaceColor;
    /*
     if (mesh->HasColors())
     surfaceColor = mesh->InterpolateTriColor(triIndex, rayHit.b1, rayHit.b2);
     else*/
    surfaceColor = Spectrum(1.f, 1.f, 1.f);
    
    // Flip the normal if required
    //Normal shadeN = (Dot(pathRay.d, N) > 0.f) ? -N : N;
    
    //------------------------------------------------------------------
    // Compute direct illumination
    //------------------------------------------------------------------
    
    Spectrum lightColor(0.f, 0.f, 0.f);
    if (triSurfMat->IsDiffuse()) {
        
        // ONE UNIFORM direct sampling light strategy
        const Spectrum lightThroughtput = surfaceColor;
        // Select the light to sample
        float lightStrategyPdf;
        const LightSource *light = scene->SampleAllLights(rnd->floatValue(),
                                                          &lightStrategyPdf);
        
        Ray shadowRay;
        float lightPdf;
        Spectrum Li = light->Sample_L(scene, hitpoint, &shadeN,
                                      rnd->floatValue(), rnd->floatValue(), rnd->floatValue(),
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
    
    return radiance;
    
}
