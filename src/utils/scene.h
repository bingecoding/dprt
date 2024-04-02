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

#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "utils/properties.h"
#include "utils/perspectivecamera.h"
#include "utils/material.h"
#include "utils/light.h"
#include "utils/texmap.h"

#include "bvhaccel.h"

class Scene {
public:
    Scene(const Properties &cfg, AcceleratorType accelType);
	~Scene();
    
    std::deque<TriangleMesh *> m_objectMeshes;
    
    std::vector<Material *> m_objectMaterials;
    std::vector<LightSource *> m_lights;
    
    TextureMapCache *m_texMapCache; // Texture maps
    std::vector<TexMapInstance *> m_objectTexMaps; // One for each object
    std::vector<BumpMapInstance *> m_objectBumpMaps; // One for each object
    std::vector<NormalMapInstance *> m_objectNormalMaps; // One for each object
    
    PerspectiveCamera *m_camera;
    
    void SetAcceleratorType(AcceleratorType type) { m_accelType = type; }
    AcceleratorType GetAcceleratorType() const { return m_accelType; }
    
    const Accelerator *GetAccelerator() const { return m_accel; }
    
    unsigned int GetTotalVertexCount() const { return m_totalVertexCount; }
	unsigned int GetTotalTriangleCount() const { return m_totalTriangleCount; }
    
    const TriangleMeshID *GetMeshIDTable() const
    {
        return m_accel->GetMeshIDTable();
    }
    
    LightSource *SampleAllLights(const float u, float *pdf) const
    {
        // One Uniform light strategy
        const unsigned int lightIndex = Min<unsigned int>(Floor2UInt(m_lights.size() * u), m_lights.size() - 1);
        
        *pdf = 1.f / m_lights.size();
        
        if(m_lights.size() == 0)
            throw std::runtime_error("No light sources available.");

        
        return m_lights[lightIndex];
    
    }
    
    LightSource *GetLightSource(const unsigned int triIndex) const
    {

        for(int i=0; i < m_lights.size(); i++) {
            if(m_lights[i]->IsAreaLight()) {
                TriangleLight *ls = (TriangleLight *)m_lights[i];
                if(ls->GetTriIndex() == triIndex)
                    return m_lights[i];
            }
        }
        
        return NULL;
        
    }

    
    bool Intersect(Ray *ray, RayHit *rayHit) {
        return m_accel->Intersect(ray, rayHit);
    }

    bool IntersectP(Ray *ray) {
        return m_accel->IntersectP(ray);
    }
    
    friend class OpenCLIntersectionDevice;
    
    BBox m_bbox;
    Accelerator *m_accel;
    
private:
    
    unsigned int m_totalVertexCount;
    unsigned int m_totalTriangleCount;
    
	BSphere m_bsphere;
    
    AcceleratorType m_accelType;
	
    TriangleMeshID Add(TriangleMesh *mesh);
    Material *CreateMaterial(const tinyobj::shape_t &shape);
    void BuildAccelerator();
};

#endif
