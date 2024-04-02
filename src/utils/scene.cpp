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

#include <iostream>

#include "utils/scene.h"
#include "utils/tiny_obj_loader.h"
#include "transform.h"

using namespace std;

Scene::Scene(const Properties &cfg, AcceleratorType accelType)
{
    m_texMapCache = new TextureMapCache();
    
    string fileName = cfg.GetString("scene.file");
    
    RT_LOG("Reading scene: " << fileName);
    
    // Camera position
    std::vector<float> vf = cfg.GetFloatVector("camera.lookat.origin");
    Point origin(vf.at(0), vf.at(1), vf.at(2));
    
    // Camera direction
    vf = cfg.GetFloatVector("camera.lookat.target");
    Point target(vf.at(0), vf.at(1), vf.at(2));
    
    // Camera up vector
    vf = cfg.GetFloatVector("camera.lookat.up");
    const Vector up(vf.at(0), vf.at(1), vf.at(2));
    
    RT_LOG("Camera position: " << origin);
	RT_LOG("Camera direction: " << target);
    RT_LOG("Camera up: " << up);
    
    m_camera = new PerspectiveCamera(origin, target, up);
    
    // Loading scene
    std::size_t found = fileName.find_last_of("/\\");
    std::string err;
    std::vector<tinyobj::shape_t> shapes;
    string basePath = "";
    if(found != std::string::npos) {
        basePath = fileName.substr(0,found+1);
        err = tinyobj::LoadObj(shapes, fileName.c_str(), basePath.c_str());
    }else {
        err = tinyobj::LoadObj(shapes, fileName.c_str());
    }
    
    if (!err.empty()) {
        RT_LOG(err);
        throw;
    }
    
    m_totalVertexCount = 0;
    m_totalTriangleCount = 0;
    
    RT_LOG("# of objects : " << shapes.size());
    
    //--------------------------------------------------------------------------
	// Build Mesh
	//--------------------------------------------------------------------------
    for (size_t i = 0; i < shapes.size(); i++) {
        
        TriangleMesh *meshObject = TriangleMesh::LoadTriangleMesh(shapes[i]);
        
        const std::string objName = shapes[i].name;
        m_objectMeshes.push_back(meshObject);
        m_totalVertexCount += meshObject->GetTotalVertexCount();
        m_totalTriangleCount += meshObject->GetTotalTriangleCount();
        m_bbox = Union(m_bbox, meshObject->GetBBox());
        m_bsphere = m_bbox.BoundingSphere();
        
        Material *material = CreateMaterial(shapes[i]);
        // Check if it is a light source
		if (material->IsLightSource()) {
			RT_LOG("The " << objName << " object is a light source with "
                   << meshObject->GetTotalTriangleCount() << " triangles");
            
			AreaLightMaterial *light = (AreaLightMaterial *)material;
			m_objectMaterials.push_back(material);
            for (unsigned int i = 0; i < meshObject->GetTotalTriangleCount(); ++i) {
                TriangleLight *tl = new TriangleLight(light,
                                                      m_objectMeshes.size() - 1,
                                                      i, m_objectMeshes);
				m_lights.push_back(tl);
            }
        } else {
            SurfaceMaterial *surfMat = (SurfaceMaterial *)material;
			m_objectMaterials.push_back(surfMat);
        }
        
        /*
        bool foundTexMap = false;
        for(int k=0; k < 3; k++) {
            std::string texMap;
            switch(k) {
                case 0:
                    texMap = shapes[i].material.diffuse_texname;
                    break;
                case 1:
                    texMap = shapes[i].material.specular_texname;
                    break;
                case 2: {
                    // Get alpha texture
                    std::map<string, string> texAlpha = shapes[i].material.unknown_parameter;
                    std::map<string, string>::iterator it = texAlpha.find("map_d");
                    if(it != texAlpha.end())
                        texMap = it->second;
                    break;
                }
                default:
                    assert(false);
            }
            
            if(texMap != ""){
                if(!meshObject->HasUVs())
                    throw std::runtime_error("Mesh object " + objName + " is missing UV coordinates for texture mapping");
                if(basePath!="")
                    texMap = basePath + texMap;
                std::replace(texMap.begin(), texMap.end(), '\\', '/');
                TexMapInstance *tm = m_texMapCache->GetTexMapInstance(texMap);
                m_objectTexMaps.push_back(tm);
                foundTexMap = true;
            }
        }
        
        if(!foundTexMap)
            m_objectTexMaps.push_back(NULL);
        */
        std::string texMap = shapes[i].material.diffuse_texname;
        if(texMap != "") {
            if(!meshObject->HasUVs())
                throw std::runtime_error("Mesh object " + objName + " is missing UV coordinates for texture mapping");
            if(basePath!="")
                texMap = basePath + texMap;
            std::replace(texMap.begin(), texMap.end(), '\\', '/');
            TexMapInstance *tm = m_texMapCache->GetTexMapInstance(texMap);
            m_objectTexMaps.push_back(tm);
            
        } else
            m_objectTexMaps.push_back(NULL);

        
        // Bump maps
        std::map<string, string> texMaps = shapes[i].material.unknown_parameter;
        std::map<string, string>::iterator it = texMaps.find("map_Bump");
        std::string bumpMap;
        if(it != texMaps.end()) {
            bumpMap = it->second;
        }
        if(bumpMap != ""){
            if(!meshObject->HasUVs())
                throw std::runtime_error("Mesh object " + objName + " is missing UV coordinates for bump mapping");
            if(basePath!="")
                bumpMap = basePath + bumpMap;
            std::replace(bumpMap.begin(), bumpMap.end(), '\\', '/');
            const float scale = 1.f;
            BumpMapInstance *bm = m_texMapCache->GetBumpMapInstance(bumpMap, scale);
            m_objectBumpMaps.push_back(bm);
        } else {
            m_objectBumpMaps.push_back(NULL);
        }
        
    }
    
    RT_LOG("Total vertex count: " << m_totalVertexCount);
    RT_LOG("Total triangle count: " << m_totalTriangleCount);
    
    if(m_totalTriangleCount == 0) {
        throw std::runtime_error("No scene data to process.");
    }
    
    //--------------------------------------------------------------------------
	// Create Acceleration Structure
	//--------------------------------------------------------------------------
    SetAcceleratorType(accelType);
    BuildAccelerator();
    
}

Material *Scene::CreateMaterial(const tinyobj::shape_t &shape) {
    
    const string matType = shape.material.name;
    if (matType.find("light") != std::string::npos) {
        
		const Spectrum gain(shape.material.ambient[0],
                            shape.material.ambient[1],
                            shape.material.ambient[2]);
        
		return new AreaLightMaterial(gain);
        
    } else if(matType.find("plastic") != std::string::npos) {
        
        const Spectrum col(shape.material.diffuse[0],
                           shape.material.diffuse[1],
                           shape.material.diffuse[2]);
        
        const Spectrum spec(shape.material.specular[0],
                            shape.material.specular[1],
                            shape.material.specular[2]);
        
        const float exp = shape.material.shininess;
        
        return new PlasticMaterial(col, spec, exp);
        
    }else if(matType.find("blinnphong") != std::string::npos) {
        
        const Spectrum col(shape.material.diffuse[0],
                           shape.material.diffuse[1],
                           shape.material.diffuse[2]);
        
        const Spectrum spec(shape.material.specular[0],
                           shape.material.specular[1],
                           shape.material.specular[2]);
        
        const float exp = shape.material.shininess;
        
        return new BlinnPhongMaterial(col, spec, exp);
        
    } else {
        
        const Spectrum col(shape.material.diffuse[0],
                           shape.material.diffuse[1],
                           shape.material.diffuse[2]);
        
        return new MatteMaterial(col);
        
    }

}

void Scene::BuildAccelerator()
{
    // Build accelerator
    switch (m_accelType) {
		case ACCEL_BVH: {
			const int treeType = 4; // Tree type to generate (2 = binary, 4 = quad, 8 = octree)
			const int costSamples = 0; // Samples for cost minimization
			const int isectCost = 80;
			const int travCost = 10;
			const float emptyBonus = 0.5f;
            
			m_accel = new BVHAccel(treeType, costSamples, isectCost,
                                   travCost, emptyBonus);
			break;
		}
        default:
            assert(false);
    }
    
    m_accel->Init(m_objectMeshes, m_totalVertexCount, m_totalTriangleCount);
    
}

Scene::~Scene()
{
    // TODO: should also delete meshes !!!
    for (std::vector<LightSource *>::const_iterator l = m_lights.begin(); l != m_lights.end(); ++l)
		delete *l;
    
    delete m_accel;

}
