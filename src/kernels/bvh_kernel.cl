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

//#pragma OPENCL EXTENSION cl_amd_printf : enable
//#pragma OPENCL EXTENSION cl_intel_printf : enable

#ifndef SKIP_RAY
#define SKIP_RAY 777.f
#endif

typedef struct {
	float x, y, z;
} Point;

typedef struct {
	float x, y, z;
} Vector;

typedef struct {
	Point o;
	Vector d;
	float mint, maxt;
} Ray;

typedef struct {
	float t;
	float b1, b2; // Barycentric coordinates of the hit point
	unsigned int index;
} RayHit;

typedef struct {
	unsigned int v[3];
} Triangle;

typedef struct {
	Point pMin, pMax;
} BBox;

typedef struct {
	BBox bbox;
	unsigned int primitive;
	unsigned int skipIndex;
} BVHAccelArrayNode;

void TriangleIntersect(
                       const float4 rayOrig,
                       const float4 rayDir,
                       const float minT,
                       float *maxT,
                       unsigned int *hitIndex,
                       float *hitB1,
                       float *hitB2,
                       const unsigned int currentIndex,
                       __global Point *verts,
                       __global Triangle *tris) {
    
	// Load triangle vertices
	__global Point *p0 = &verts[tris[currentIndex].v[0]];
	__global Point *p1 = &verts[tris[currentIndex].v[1]];
	__global Point *p2 = &verts[tris[currentIndex].v[2]];
    
	float4 v0 = (float4) (p0->x, p0->y, p0->z, 0.f);
	float4 v1 = (float4) (p1->x, p1->y, p1->z, 0.f);
	float4 v2 = (float4) (p2->x, p2->y, p2->z, 0.f);
    
	// Calculate intersection
	float4 e1 = v1 - v0;
	float4 e2 = v2 - v0;
	float4 s1 = cross(rayDir, e2);
    
	const float divisor = dot(s1, e1);
	if (divisor == 0.f)
		return;
    
	const float invDivisor = 1.f / divisor;
    
	// Compute first barycentric coordinate
	const float4 d = rayOrig - v0;
	const float b1 = dot(d, s1) * invDivisor;
	if (b1 < 0.f)
		return;
    
	// Compute second barycentric coordinate
	const float4 s2 = cross(d, e1);
	const float b2 = dot(rayDir, s2) * invDivisor;
	if (b2 < 0.f)
		return;
    
	const float b0 = 1.f - b1 - b2;
	if (b0 < 0.f)
		return;
    
	// Compute _t_ to intersection point
	const float t = dot(e2, s2) * invDivisor;
	if (t < minT || t > *maxT)
		return;
    
	*maxT = t;
	*hitB1 = b1;
	*hitB2 = b2;
	*hitIndex = currentIndex;
}

int BBoxIntersectP(
                   const float4 rayOrig, const float4 invRayDir,
                   const float mint, const float maxt,
                   const float4 pMin, const float4 pMax) {
	const float4 l1 = (pMin - rayOrig) * invRayDir;
	const float4 l2 = (pMax - rayOrig) * invRayDir;
	const float4 tNear = fmin(l1, l2);
	const float4 tFar = fmax(l1, l2);
    
	float t0 = max(max(max(tNear.x, tNear.y), max(tNear.x, tNear.z)), mint);
    float t1 = min(min(min(tFar.x, tFar.y), min(tFar.x, tFar.z)), maxt);
    
	return (t1 > t0);
}

__kernel __attribute__((work_group_size_hint(64, 1, 1)))
void Intersect(
                        __global Ray *rays,
                        __global RayHit *rayHits,
                        __global Point *verts,
                        __global Triangle *tris,
                        const unsigned int triangleCount,
                        const unsigned int nodeCount,
                        __global BVHAccelArrayNode *bvhTree,
                        const unsigned int rayCount) {
	// Select the ray to check
	const int gid = get_global_id(0);
	if (gid >= rayCount)
		return;
    
	float4 rayOrig,rayDir;
	float minT, maxT;
	{
		__global float4 *basePtr =(__global float4 *)&rays[gid];
		float4 data0 = (*basePtr++);
		float4 data1 = (*basePtr);
        
		rayOrig = (float4)(data0.x, data0.y, data0.z, 0.f);
		rayDir = (float4)(data0.w, data1.x, data1.y, 0.f);
        
		minT = data1.z;
		maxT = data1.w;
	}
    
    if(maxT == SKIP_RAY){
        return;
    }
    
	//float4 rayOrig = (float4) (rays[gid].o.x, rays[gid].o.y, rays[gid].o.z, 0.f);
	//float4 rayDir = (float4) (rays[gid].d.x, rays[gid].d.y, rays[gid].d.z, 0.f);
	//float minT = rays[gid].mint;
	//float maxT = rays[gid].maxt;
    
	float4 invRayDir = (float4) 1.f / rayDir;
    
	unsigned int hitIndex = 0xffffffffu;
	unsigned int currentNode = 0; // Root Node
	float b1, b2;
	unsigned int stopNode = bvhTree[0].skipIndex; // Non-existent
    
	float4 pMin, pMax, data0, data1;
	__global float4 *basePtr;
	while (currentNode < stopNode) {
		/*float4 pMin = (float4)(bvhTree[currentNode].bbox.pMin.x,
         bvhTree[currentNode].bbox.pMin.y,
         bvhTree[currentNode].bbox.pMin.z,
         0.f);
         float4 pMax = (float4)(bvhTree[currentNode].bbox.pMax.x,
         bvhTree[currentNode].bbox.pMax.y,
         bvhTree[currentNode].bbox.pMax.z,
         0.f);*/
        
		basePtr =(__global float4 *)&bvhTree[currentNode];
		data0 = (*basePtr++);
		data1 = (*basePtr);
        
		pMin = (float4)(data0.x, data0.y, data0.z, 0.f);
		pMax = (float4)(data0.w, data1.x, data1.y, 0.f);
        
		if (BBoxIntersectP(rayOrig, invRayDir, minT, maxT, pMin, pMax)) {
			//const unsigned int triIndex = bvhTree[currentNode].primitive;
			const unsigned int triIndex = as_uint(data1.z);
            
			if (triIndex != 0xffffffffu)
				TriangleIntersect(rayOrig, rayDir, minT, &maxT, &hitIndex, &b1, &b2, triIndex, verts, tris);
            
			currentNode++;
		} else {
			//bvhTree[currentNode].skipIndex;
			currentNode = as_uint(data1.w);
		}
	}
    
	// Write result
	rayHits[gid].t = maxT;
	rayHits[gid].b1 = b1;
	rayHits[gid].b2 = b2;
	rayHits[gid].index = hitIndex;
    
}