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

#include "utils/light.h"
#include "utils/scene.h"
#include "utils/montecarlo.h"

TriangleLight::TriangleLight(const AreaLightMaterial *mat,
                             const unsigned int mshIndex,
                             const unsigned int triangleIndex,
                             const std::deque<TriangleMesh *> &objs)
{
	m_lightMaterial = mat;
	m_meshIndex = mshIndex;
	m_triIndex = triangleIndex;
    
	Init(objs);
}

void TriangleLight::Init(const std::deque<TriangleMesh *> &objs)
{
	const TriangleMesh *mesh = objs[m_meshIndex];
	m_area = mesh->GetTriangleArea(m_triIndex);
}

Spectrum TriangleLight::Sample_L(const Scene *scene, const Point &p,
                                 const Normal *N, const float u0,
                                 const float u1, const float u2, float *pdf,
                                 Ray *shadowRay) const
{
	const TriangleMesh *mesh = scene->m_objectMeshes[m_meshIndex];
    
	Point samplePoint;
	float b0, b1, b2;
	mesh->Sample(m_triIndex, u0, u1, &samplePoint, &b0, &b1, &b2);
	const Normal &sampleN = mesh->GetNormal(m_triIndex, 0);
    
	Vector wi = samplePoint - p;
	const float distanceSquared = wi.LengthSquared();
	const float distance = sqrtf(distanceSquared);
	wi /= distance;
    
	const float sampleNdotMinusWi = Dot(sampleN, -wi);
	if ((sampleNdotMinusWi <= 0.f) || (N && Dot(*N, wi) <= 0.f)) {
		*pdf = 0.f;
		return Spectrum();
	}
    
	*shadowRay = Ray(p, wi, RAY_EPSILON, distance - RAY_EPSILON);
	*pdf = distanceSquared / (sampleNdotMinusWi * m_area);
    
	// Using 0.1 instead of 0.0 to cut down fireflies
	if (*pdf <= 0.1f) {
		*pdf = 0.f;
		return Spectrum();
	}
    
    if (mesh->HasColors())
        return mesh->GetColor(m_triIndex) * m_lightMaterial->GetGain(); // Light sources are supposed to have flat color
    else
        return m_lightMaterial->GetGain(); // Light sources are supposed to have flat color

}

Spectrum TriangleLight::Sample_L(const Scene *scene, const float u0,
                                 const float u1, const float u2, const float u3,
                                 const float u4, float *pdf, Ray *ray,
                                 Normal *Nl) const
{
	const TriangleMesh *mesh = scene->m_objectMeshes[m_meshIndex];
    
	// Ray origin
	float b0, b1, b2;
	Point orig;
	mesh->Sample(m_triIndex, u0, u1, &orig, &b0, &b1, &b2);
    
	// Ray direction
	const Normal &sampleN = mesh->GetNormal(m_triIndex, 0); // Light sources are supposed to be flat
    if(Nl != NULL) {
        *Nl = sampleN;
    }
	Vector dir = UniformSampleSphere(u2, u3);
	float RdotN = Dot(dir, sampleN);
	if (RdotN < 0.f) {
		dir *= -1.f;
		RdotN = -RdotN;
	}
    
	*ray = Ray(orig, dir);
    
	*pdf = INV_TWOPI / m_area;

    if (mesh->HasColors())
        return mesh->GetColor(m_triIndex) * m_lightMaterial->GetGain() * RdotN; // Light sources are supposed to have flat color
    else
        return m_lightMaterial->GetGain() * RdotN; // Light sources are supposed to have flat color

}
