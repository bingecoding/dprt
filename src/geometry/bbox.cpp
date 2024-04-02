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

#include "geometry/bbox.h"

// BBox Method Definitions

BBox Union(const BBox &b, const Point &p) {
	BBox ret = b;
	ret.pMin.x = Min(b.pMin.x, p.x);
	ret.pMin.y = Min(b.pMin.y, p.y);
	ret.pMin.z = Min(b.pMin.z, p.z);
	ret.pMax.x = Max(b.pMax.x, p.x);
	ret.pMax.y = Max(b.pMax.y, p.y);
	ret.pMax.z = Max(b.pMax.z, p.z);
	return ret;
}

BBox Union(const BBox &b, const BBox &b2) {
	BBox ret;
	ret.pMin.x = Min(b.pMin.x, b2.pMin.x);
	ret.pMin.y = Min(b.pMin.y, b2.pMin.y);
	ret.pMin.z = Min(b.pMin.z, b2.pMin.z);
	ret.pMax.x = Max(b.pMax.x, b2.pMax.x);
	ret.pMax.y = Max(b.pMax.y, b2.pMax.y);
	ret.pMax.z = Max(b.pMax.z, b2.pMax.z);
	return ret;
}

Point ClosestPointToBBox(Point p, BBox box)
{
    Point q;
    Point v;
    v.x = p.x;
    if(v.x < box.pMin.x)
        v.x = box.pMin.x;
    if(v.x > box.pMax.x)
        v.x = box.pMax.x;
    
    q.x = v.x;
    
    v.y = p.y;
    if(v.y < box.pMin.y)
        v.y = box.pMin.y;
    if(v.y > box.pMax.y)
        v.y = box.pMax.y;
    
    q.y = v.y;
    
    v.z = p.z;
    if(v.z < box.pMin.z)
        v.z = box.pMin.z;
    if(v.z > box.pMax.z)
        v.z = box.pMax.z;
    
    q.z = v.z;
    
    return q;
}

void BBox::BoundingSphere(Point *c, float *rad) const {
	*c = .5f * (pMin + pMax);
	*rad = Inside(*c) ? Distance(*c, pMax) : 0.f;
}

BSphere BBox::BoundingSphere() const {
	const Point c = .5f * (pMin + pMax);
	const float rad = Inside(c) ? Distance(c, pMax) : 0.f;
    
	return BSphere(c, rad);
}

// NOTE - lordcrc - BBox::IntersectP relies on IEEE 754 behaviour of infinity and /fp:fast breaks this
#if defined(WIN32) && !defined(__CYGWIN__)
#pragma float_control(push)
#pragma float_control(precise, on)
#endif
bool BBox::IntersectP(const Ray &ray, float *hitt0,
                      float *hitt1) const {
	float t0 = ray.mint, t1 = ray.maxt;
	for (int i = 0; i < 3; ++i) {
        
		// Update interval for ith bounding box slab
		float invRayDir = 1.f / ray.d[i];
		float tNear = (pMin[i] - ray.o[i]) * invRayDir;
		float tFar = (pMax[i] - ray.o[i]) * invRayDir;
		
        // Update parametric interval from slab intersection t
		if (tNear > tFar) {
            Swap(tNear, tFar);
        }
		
        t0 = tNear > t0 ? tNear : t0;
		t1 = tFar < t1 ? tFar : t1;
		if (t0 > t1) {
            return false;
        }
	}
    
	if (hitt0) {
        *hitt0 = t0;
    }
    
	if (hitt1) {
        *hitt1 = t1;
    }
	
    return true;
}
#if defined(WIN32) && !defined(__CYGWIN__)
#pragma float_control(pop)
#endif
