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

#ifndef LIGHT_H
#define LIGHT_H

#include "utils/trianglemesh.h"
#include "utils/material.h"

class Scene;

enum LightSourceType {
    TYPE_TRIANGLELIGHT
};

class LightSource {
public:
    
    virtual ~LightSource() {}
    
    virtual LightSourceType GetType() const=0;
    
    virtual bool IsAreaLight() const { return false; }
    
	virtual Spectrum Sample_L(const Scene *scene, const Point &p,
                              const Normal *N, const float u0, const float u1,
                              const float u2, float *pdf,
                              Ray *shadowRay) const = 0;
    
	virtual Spectrum Sample_L(const Scene *scene, const float u0,
                              const float u1, const float u2, const float u3,
                              const float u4, float *pdf, Ray *ray, Normal *Nl = NULL) const = 0;
};

class TriangleLight : public LightSource {
public:
	TriangleLight() { }
	TriangleLight(const AreaLightMaterial *mat, const unsigned int mshIndex,
                  const unsigned int triangleIndex,
                  const std::deque<TriangleMesh *> &objs);
    
	bool IsAreaLight() const { return true; }
    
	LightSourceType GetType() const { return TYPE_TRIANGLELIGHT; }
    
	void SetMaterial(const AreaLightMaterial *mat) { m_lightMaterial = mat; }
	const Material *GetMaterial() const { return m_lightMaterial; }
    
	Spectrum Sample_L(const Scene *scene, const Point &p, const Normal *N,
                      const float u0, const float u1, const float u2,
                      float *pdf, Ray *shadowRay) const;
	Spectrum Sample_L(const Scene *scene,
                      const float u0, const float u1, const float u2,
                      const float u3, const float u4, float *pdf,
                      Ray *ray, Normal *Nl = NULL) const;
    
	void Init(const std::deque<TriangleMesh *> &objs);
	unsigned int GetMeshIndex() const { return m_meshIndex; }
	unsigned int GetTriIndex() const { return m_triIndex; }
	float GetArea() const { return m_area; }
    
private:
	const AreaLightMaterial *m_lightMaterial;
	unsigned int m_meshIndex, m_triIndex;
	float m_area;
};


#endif
