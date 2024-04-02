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

//#pragma OPENCL EXTENSION cl_intel_printf : enable

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef INV_PI
#define INV_PI  0.31830988618379067154f
#endif

#ifndef INV_TWOPI
#define INV_TWOPI  0.15915494309189533577f
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------

typedef struct {
	float r, g, b;
} Spectrum;

typedef struct {
	float x, y, z;
} Point;

typedef struct {
	float x, y, z;
} Vector;

typedef struct {
	uint v0, v1, v2;
} Triangle;

typedef struct {
	Point o;
	Vector d;
	float mint, maxt;
} Ray;

typedef struct {
	float t;
	float b1, b2; // Barycentric coordinates of the hit point
	uint index;
} RayHit;

typedef struct {
	uint s1, s2, s3;
} Seed;

#define PATH_STATE_NEXT_VERTEX 0
#define PATH_STATE_SAMPLE_LIGHT 1
#define PATH_STATE_CONTINUE_NEXT_VERTEX 2
#define PATH_STATE_DIRECT_LIGHT 3
#define PATH_STATE_VPL_LIGHT 4
#define PATH_STATE_CAMERA_RAY 5
#define PATH_STATE_END 6

typedef struct {
	float3 throughput;
	uint depth, pixelIndex, subpixelIndex;
    uint lightCount;
	Seed seed;
    
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	int specularBounce;
	int state;
    
	Ray pathRay;
	RayHit pathHit;
	float3 lightRadiance;
	float3 accumRadiance;
#endif
} Path;

typedef struct {
	   
	uint depth;
    int terminatePath;
	Seed seed;

	int specularBounce;
	int state;
    
	Ray pathRay;
	RayHit pathHit;
    Point hitPoint;
    
    float pdf;
    Spectrum contrib;
	Spectrum alpha;
    Vector n;
    
} VPL;

typedef struct {
	Spectrum c;
	uint count;
} Pixel;

#define MAT_MATTE 0
#define MAT_AREALIGHT 1
#define MAT_MIRROR 2
#define MAT_GLASS 3
#define MAT_MATTEMIRROR 4
#define MAT_METAL 5
#define MAT_MATTEMETAL 6
#define MAT_BLINNPHONG 7
#define MAT_ARCHGLASS 8
#define MAT_ALLOY 9
#define MAT_NULL 10

typedef struct {
    float r, g, b;
} MatteParam;

typedef struct {
    MatteParam matte;
    float spec_r, spec_g, spec_b;
    float exponent;
} BlinnPhongParam;

typedef struct {
    float gain_r, gain_g, gain_b;
    int specularBounce;
} AreaLightParam;

typedef struct {
    float r, g, b;
    int specularBounce;
} MirrorParam;

typedef struct {
    float refl_r, refl_g, refl_b;
    float refrct_r, refrct_g, refrct_b;
    float ousideIor, ior;
    float R0;
    int reflectionSpecularBounce, transmitionSpecularBounce;
} GlassParam;

typedef struct {
	MatteParam matte;
	MirrorParam mirror;
	float matteFilter, totFilter, mattePdf, mirrorPdf;
} MatteMirrorParam;

typedef struct {
    float r, g, b;
    float exponent;
    int specularBounce;
} MetalParam;

typedef struct {
	MatteParam matte;
	MetalParam metal;
	float matteFilter, totFilter, mattePdf, metalPdf;
} MatteMetalParam;

typedef struct {
    float diff_r, diff_g, diff_b;
    float refl_r, refl_g, refl_b;
    float exponent;
    float R0;
    int specularBounce;
} AlloyParam;

typedef struct {
    float refl_r, refl_g, refl_b;
    float refrct_r, refrct_g, refrct_b;
	float transFilter, totFilter, reflPdf, transPdf;
	bool reflectionSpecularBounce, transmitionSpecularBounce;
} ArchGlassParam;

typedef struct {
	unsigned int type;
	union {
		MatteParam matte;
        AreaLightParam areaLight;
		MirrorParam mirror;
        GlassParam glass;
		MatteMirrorParam matteMirror;
        MetalParam metal;
        MatteMetalParam matteMetal;
        AlloyParam alloy;
        ArchGlassParam archGlass;
	} param;
} Material;

typedef struct {
	Point v0, v1, v2;
	Vector normal;
	float area;
	float gain_r, gain_g, gain_b;
} TriangleLight;

typedef struct {
    float u, v;
} UV;

typedef struct {
    unsigned int rgbOffset, alphaOffset;
    unsigned int width, height;
} TexMap;

//------------------------------------------------------------------------------
// Ray intersection code to trace ray when evaluating the root node, this
// is necessary because the representative light is the same as one of its
// children; this means we can reuse the material term and trace less rays.
//------------------------------------------------------------------------------

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
    __global Point *p0 = &verts[tris[currentIndex].v0];
    __global Point *p1 = &verts[tris[currentIndex].v1];
    __global Point *p2 = &verts[tris[currentIndex].v2];
    
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

void Intersect(__local Ray *ray, __local RayHit *rayHit,
               __global Point *verts, __global Triangle *tris,
               __global BVHAccelArrayNode *bvhTree)
{
    // Only trace one ray
    float4 rayOrig,rayDir;
    float minT, maxT;
    {
        __local float4 *basePtr =(__local float4 *)ray;
        float4 data0 = (*basePtr++);
        float4 data1 = (*basePtr);
        
        rayOrig = (float4)(data0.x, data0.y, data0.z, 0.f);
        rayDir = (float4)(data0.w, data1.x, data1.y, 0.f);
        
        minT = data1.z;
        maxT = data1.w;
    }
    
    float4 invRayDir = (float4) 1.f / rayDir;
    
    unsigned int hitIndex = 0xffffffffu;
    unsigned int currentNode = 0; // Root Node
    float b1, b2;
    unsigned int stopNode = bvhTree[0].skipIndex; // Non-existent
    
    float4 pMin, pMax, data0, data1;
    __global float4 *basePtr;
    while (currentNode < stopNode) {
        
        basePtr =(__global float4 *)&bvhTree[currentNode];
        data0 = (*basePtr++);
        data1 = (*basePtr);
        
        pMin = (float4)(data0.x, data0.y, data0.z, 0.f);
        pMax = (float4)(data0.w, data1.x, data1.y, 0.f);
        
        if (BBoxIntersectP(rayOrig, invRayDir, minT, maxT, pMin, pMax)) {
            //const unsigned int triIndex = bvhTree[currentNode].primitive;
            const unsigned int triIndex = as_uint(data1.z);
            
            if (triIndex != 0xffffffffu)
                TriangleIntersect(rayOrig, rayDir, minT, &maxT, &hitIndex, &b1,
                                  &b2, triIndex, verts, tris);
            
            currentNode++;
        } else {
            //bvhTree[currentNode].skipIndex;
            currentNode = as_uint(data1.w);
        }
    }
    
    // Write result
    rayHit->t = maxT;
    rayHit->b1 = b1;
    rayHit->b2 = b2;
    rayHit->index = hitIndex;
    
}

//------------------------------------------------------------------------------
// Random number generator
// maximally equidistributed combined Tausworthe generator
//------------------------------------------------------------------------------

#define FLOATMASK 0x00ffffffu

uint TAUSWORTHE(const uint s, const uint a,
                const uint b, const uint c,
                const uint d) {
	return ((s&c)<<d) ^ (((s << a) ^ s) >> b);
}

uint LCG(const uint x) { return x * 69069; }

uint ValidSeed(const uint x, const uint m) {
	return (x < m) ? (x + m) : x;
}

void InitRandomGenerator(uint seed, Seed *s) {
	// Avoid 0 value
	seed = (seed == 0) ? (seed + 0xffffffu) : seed;
    
	s->s1 = ValidSeed(LCG(seed), 1);
	s->s2 = ValidSeed(LCG(s->s1), 7);
	s->s3 = ValidSeed(LCG(s->s2), 15);
}

unsigned long RndUintValue(Seed *s) {
	s->s1 = TAUSWORTHE(s->s1, 13, 19, 4294967294UL, 12);
	s->s2 = TAUSWORTHE(s->s2, 2, 25, 4294967288UL, 4);
	s->s3 = TAUSWORTHE(s->s3, 3, 11, 4294967280UL, 17);
    
	return ((s->s1) ^ (s->s2) ^ (s->s3));
}

float RndFloatValue(Seed *s) {
	return (RndUintValue(s) & FLOATMASK) * (1.f / (FLOATMASK + 1UL));
}

float Y(const float3 color) {
    return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
}

void LVSTORE3F(const float3 v, __local float *p) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
}

float3 VLOAD3F(const __global float *p) {
    return (float3)(p[0], p[1], p[2]);
}

void VSTORE3F(const float3 v, __global float *p) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
}

void Ray_Init4_Private(Ray *ray, const float3 orig, const float3 dir,
                       const float mint, const float maxt) {
    ray->o.x = orig.x;
    ray->o.y = orig.y;
    ray->o.z = orig.z;
    
    ray->d.x = dir.x;
    ray->d.y = dir.y;
    ray->d.z = dir.z;
    
    ray->mint = mint;
    ray->maxt = maxt;
}

void Ray_Init4_Local(__local Ray *ray, const float3 orig, const float3 dir,
                     const float mint, const float maxt) {
    LVSTORE3F(orig, &ray->o.x);
    LVSTORE3F(dir, &ray->d.x);
    ray->mint = mint;
    ray->maxt = maxt;
}

void Ray_Init4(__global Ray *ray, const float3 orig, const float3 dir,
               const float mint, const float maxt) {
    VSTORE3F(orig, &ray->o.x);
    VSTORE3F(dir, &ray->d.x);
    ray->mint = mint;
    ray->maxt = maxt;
}


//------------------------------------------------------------------------------

float3 Triangle_InterpolateNormal(const float3 n0, const float3 n1, const float3 n2,
                                  const float b0, const float b1, const float b2) {
    return normalize(b0 * n0 + b1 * n1 + b2 * n2);
}

float3 Triangle_InterpolateColor(const float3 rgb0, const float3 rgb1, const float3 rgb2,
                                 const float b0, const float b1, const float b2) {
    return b0 * rgb0 + b1 * rgb1 + b2 * rgb2;
}

float3 UniformSampleSphere(float u1, float u2)
{
    float z = 1.f - 2.f * u1;
    float r = sqrt(max(0.f, 1.f - z*z));
    float phi = 2.f * M_PI * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    
    return (float3)(x, y, z);
}

void ConcentricSampleDisk(const float u1, const float u2, float *dx, float *dy) {
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2.f * u1 - 1.f;
	float sy = 2.f * u2 - 1.f;
	// Map square to $(r,\theta)$
	// Handle degeneracy at the origin
	if (sx == 0.f && sy == 0.f) {
		*dx = 0.f;
		*dy = 0.f;
		return;
	}
	if (sx >= -sy) {
		if (sx > sy) {
			// Handle first region of disk
			r = sx;
			if (sy > 0.f)
				theta = sy / r;
			else
				theta = 8.f + sy / r;
		} else {
			// Handle second region of disk
			r = sy;
			theta = 2.f - sx / r;
		}
	} else {
		if (sx <= sy) {
			// Handle third region of disk
			r = -sx;
			theta = 4.f - sy / r;
		} else {
			// Handle fourth region of disk
			r = -sy;
			theta = 6.f + sx / r;
		}
	}
	theta *= M_PI / 4.f;
	*dx = r * cos(theta);
	*dy = r * sin(theta);
}

float3 CosineSampleHemisphere(const float u1, const float u2) {
    float dx,dy;
    float3 dir;
	ConcentricSampleDisk(u1, u2, &dx, &dy);
    dir.x = dx;
    dir.y = dy;
    dir.z = sqrt(max(0.f, 1.f - dx * dx - dy * dy));
    return dir;
}

void CoordinateSystem(const float3 *v1, float3 *v2, float3 *v3) {
	if (fabs(v1->x) > fabs(v1->y)) {
		float invLen = 1.f / sqrt(v1->x * v1->x + v1->z * v1->z);
		v2->x = -v1->z * invLen;
		v2->y = 0.f;
		v2->z = v1->x * invLen;
	} else {
		float invLen = 1.f / sqrt(v1->y * v1->y + v1->z * v1->z);
		v2->x = 0.f;
		v2->y = v1->z * invLen;
		v2->z = -v1->y * invLen;
	}
    
    *v3 = cross(*v1, *v2);
}

//------------------------------------------------------------------------------

void GenerateRay(
                 const uint pixelIndex,
                 __global Ray *ray, Seed *seed
#if defined(PARAM_CAMERA_DYNAMIC)
                 , __global float *cameraData
#endif
                 ) {
    
	/*// Gaussina distribution
     const float rad = filterRadius * sqrt(-log(1.f - RndFloatValue(seed)));
     const float angle = 2 * M_PI * RndFloatValue(seed);
     
     const float screenX = pixelIndex % PARAM_IMAGE_WIDTH + rad * cos(angle);
     const float screenY = pixelIndex / PARAM_IMAGE_WIDTH + rad * sin(angle);*/
    
	const float screenX = pixelIndex % PARAM_IMAGE_WIDTH + RndFloatValue(seed) - 0.5f;
	const float screenY = pixelIndex / PARAM_IMAGE_WIDTH + RndFloatValue(seed) - 0.5f;
    
	float3 Pras;
	Pras.x = screenX;
	Pras.y = PARAM_IMAGE_HEIGHT - screenY - 1.f;
	Pras.z = 0;
    
	float3 orig;
	// RasterToCamera(Pras, &orig);
	const float iw = 1.f / (PARAM_RASTER2CAMERA_30 * Pras.x + PARAM_RASTER2CAMERA_31 * Pras.y + PARAM_RASTER2CAMERA_32 * Pras.z + PARAM_RASTER2CAMERA_33);
	orig.x = (PARAM_RASTER2CAMERA_00 * Pras.x + PARAM_RASTER2CAMERA_01 * Pras.y + PARAM_RASTER2CAMERA_02 * Pras.z + PARAM_RASTER2CAMERA_03) * iw;
	orig.y = (PARAM_RASTER2CAMERA_10 * Pras.x + PARAM_RASTER2CAMERA_11 * Pras.y + PARAM_RASTER2CAMERA_12 * Pras.z + PARAM_RASTER2CAMERA_13) * iw;
	orig.z = (PARAM_RASTER2CAMERA_20 * Pras.x + PARAM_RASTER2CAMERA_21 * Pras.y + PARAM_RASTER2CAMERA_22 * Pras.z + PARAM_RASTER2CAMERA_23) * iw;
    
	float3 dir = orig;
    
#if defined(PARAM_CAMERA_HAS_DOF)
	// Sample point on lens
	float lensU, lensV;
	ConcentricSampleDisk(RndFloatValue(seed), RndFloatValue(seed), &lensU, &lensV);
	lensU *= PARAM_CAMERA_LENS_RADIUS;
	lensV *= PARAM_CAMERA_LENS_RADIUS;
    
	// Compute point on plane of focus
	const float ft = (PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / dir.z;
	float3 Pfocus = orig + dir*ft;
    
	// Update ray for effect of lens
	orig.x += lensU * ((PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / PARAM_CAMERA_FOCAL_DISTANCE);
	orig.y += lensV * ((PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / PARAM_CAMERA_FOCAL_DISTANCE);
    
    dir = Pfocus - orig;
    
#endif
    dir = normalize(dir);
    
	// CameraToWorld(*ray, ray);
	float3 torig;
	const float iw2 = 1.f / (PARAM_CAMERA2WORLD_30 * orig.x + PARAM_CAMERA2WORLD_31 * orig.y + PARAM_CAMERA2WORLD_32 * orig.z + PARAM_CAMERA2WORLD_33);
	torig.x = (PARAM_CAMERA2WORLD_00 * orig.x + PARAM_CAMERA2WORLD_01 * orig.y + PARAM_CAMERA2WORLD_02 * orig.z + PARAM_CAMERA2WORLD_03) * iw2;
	torig.y = (PARAM_CAMERA2WORLD_10 * orig.x + PARAM_CAMERA2WORLD_11 * orig.y + PARAM_CAMERA2WORLD_12 * orig.z + PARAM_CAMERA2WORLD_13) * iw2;
	torig.z = (PARAM_CAMERA2WORLD_20 * orig.x + PARAM_CAMERA2WORLD_21 * orig.y + PARAM_CAMERA2WORLD_22 * orig.z + PARAM_CAMERA2WORLD_23) * iw2;
    
	float3 tdir;
	tdir.x = PARAM_CAMERA2WORLD_00 * dir.x + PARAM_CAMERA2WORLD_01 * dir.y + PARAM_CAMERA2WORLD_02 * dir.z;
	tdir.y = PARAM_CAMERA2WORLD_10 * dir.x + PARAM_CAMERA2WORLD_11 * dir.y + PARAM_CAMERA2WORLD_12 * dir.z;
	tdir.z = PARAM_CAMERA2WORLD_20 * dir.x + PARAM_CAMERA2WORLD_21 * dir.y + PARAM_CAMERA2WORLD_22 * dir.z;
    
    Ray_Init4(ray, torig, tdir, PARAM_RAY_EPSILON, (PARAM_CLIP_YON - PARAM_CLIP_HITHER) / dir.z);
}

//------------------------------------------------------------------------------

float3 Mesh_InterpolateColor(__global Spectrum *vertCols, __global Triangle *triangles,
                             const uint triIndex, const float b1, const float b2) {
    __global Triangle *tri = &triangles[triIndex];
    const float3 rgb0 = VLOAD3F(&vertCols[tri->v0].r);
    const float3 rgb1 = VLOAD3F(&vertCols[tri->v1].r);
    const float3 rgb2 = VLOAD3F(&vertCols[tri->v2].r);
    
    const float b0 = 1.f - b1 - b2;
    return Triangle_InterpolateColor(rgb0, rgb1, rgb2, b0, b1, b2);
}

float3 Mesh_InterpolateNormal(__global Vector *normals, __global Triangle *triangles,
                              const uint triIndex, const float b1, const float b2) {
    __global Triangle *tri = &triangles[triIndex];
    const float3 n0 = VLOAD3F(&normals[tri->v0].x);
    const float3 n1 = VLOAD3F(&normals[tri->v1].x);
    const float3 n2 = VLOAD3F(&normals[tri->v2].x);
    
    const float b0 = 1.f - b1 - b2;
    return Triangle_InterpolateNormal(n0, n1, n2, b0, b1, b2);
}

void Mesh_InterpolateUV(__global UV *uvs, __global Triangle *triangles,
                        const uint triIndex, const float b1, const float b2, UV *uv) {
    __global Triangle *tri = &triangles[triIndex];
    
    const float b0 = 1.f - b1 - b2;
    uv->u = b0 * uvs[tri->v0].u + b1 * uvs[tri->v1].u + b2 * uvs[tri->v2].u;
    uv->v = b0 * uvs[tri->v0].v + b1 * uvs[tri->v1].v + b2 * uvs[tri->v2].v;
}

//------------------------------------------------------------------------------

int Mod(int a, int b) {
    if (b == 0)
        b = 1;
    
    a %= b;
    if (a < 0)
        a += b;
    
    return a;
}

void TexMap_GetTexel(__global Spectrum *pixels, const uint width, const uint height,
                     const int s, const int t, Spectrum *col) {
    const uint u = Mod(s, width);
    const uint v = Mod(t, height);
    
    const unsigned index = v * width + u;
    
    col->r = pixels[index].r;
    col->g = pixels[index].g;
    col->b = pixels[index].b;
}

float TexMap_GetAlphaTexel(__global float *alphas, const uint width, const uint height,
                           const int s, const int t) {
    const uint u = Mod(s, width);
    const uint v = Mod(t, height);
    
    const unsigned index = v * width + u;
    
    return alphas[index];
}

void TexMap_GetColor(__global Spectrum *pixels, const uint width, const uint height,
                     const float u, const float v, Spectrum *col) {
    const float s = u * width - 0.5f;
    const float t = v * height - 0.5f;
    
    const int s0 = (int)floor(s);
    const int t0 = (int)floor(t);
    
    const float ds = s - s0;
    const float dt = t - t0;
    
    const float ids = 1.f - ds;
    const float idt = 1.f - dt;
    
    Spectrum c0, c1, c2, c3;
    TexMap_GetTexel(pixels, width, height, s0, t0, &c0);
    TexMap_GetTexel(pixels, width, height, s0, t0 + 1, &c1);
    TexMap_GetTexel(pixels, width, height, s0 + 1, t0, &c2);
    TexMap_GetTexel(pixels, width, height, s0 + 1, t0 + 1, &c3);
    
    const float k0 = ids * idt;
    const float k1 = ids * dt;
    const float k2 = ds * idt;
    const float k3 = ds * dt;
    
    col->r = k0 * c0.r + k1 * c1.r + k2 * c2.r + k3 * c3.r;
    col->g = k0 * c0.g + k1 * c1.g + k2 * c2.g + k3 * c3.g;
    col->b = k0 * c0.b + k1 * c1.b + k2 * c2.b + k3 * c3.b;
}

float TexMap_GetAlpha(__global float *alphas, const uint width, const uint height,
                      const float u, const float v) {
    const float s = u * width - 0.5f;
    const float t = v * height - 0.5f;
    
    const int s0 = (int)floor(s);
    const int t0 = (int)floor(t);
    
    const float ds = s - s0;
    const float dt = t - t0;
    
    const float ids = 1.f - ds;
    const float idt = 1.f - dt;
    
    const float c0 = TexMap_GetAlphaTexel(alphas, width, height, s0, t0);
    const float c1 = TexMap_GetAlphaTexel(alphas, width, height, s0, t0 + 1);
    const float c2 = TexMap_GetAlphaTexel(alphas, width, height, s0 + 1, t0);
    const float c3 = TexMap_GetAlphaTexel(alphas, width, height, s0 + 1, t0 + 1);
    
    const float k0 = ids * idt;
    const float k1 = ids * dt;
    const float k2 = ds * idt;
    const float k3 = ds * dt;
    
    return k0 * c0 + k1 * c1 + k2 * c2 + k3 * c3;
}


//------------------------------------------------------------------------------
// Materials
//------------------------------------------------------------------------------

float3 Matte_Sample_f(__global MatteParam *mat, const float3 *wio,
                    float *pdf, Spectrum *f, const float3 *shadeN,
                    const float u0, const float u1
                    )
{
	float3 dir = CosineSampleHemisphere(u0, u1);
	*pdf = dir.z * INV_PI;
    
	float3 v1, v2;
	CoordinateSystem(shadeN, &v1, &v2);
    
    float3 womega = v1*dir + v2*dir + (*shadeN)*dir;
    
	const float dp = dot(*shadeN, womega);
	// Using 0.0001 instead of 0.0 to cut down fireflies
	if (dp <= 0.0001f)
		*pdf = 0.f;
	else {
		*pdf /=  dp;
        
		f->r = mat->r * INV_PI;
		f->g = mat->g * INV_PI;
		f->b = mat->b * INV_PI;
	}
    
}

float3 Matte_f(__global MatteParam *mat, const float3 *wo,
               const float3 *wi, const float3 *shadeN)
{
    float3 col;
    col.x = mat->r;
    col.y = mat->g;
    col.z = mat->b;
    col *= INV_PI;
    
    return col;
}

float3 BlinnPhong_f(__global BlinnPhongParam *mat, const float3 *wo,
                    const float3 *wi, const float3 *shadeN)
{
    float cosThetaO = fabs(wo->z);
    float cosThetaI = fabs(wi->z);
    //if (cosThetaI == 0.f || cosThetaO == 0.f) return 0.f;
    float3 wh = *wi + *wo;
    //if (wh.x == 0. && wh.y == 0. && wh.z == 0.) return 0.f;
    wh = normalize(wh);
    
    float specAngle = max(dot(wh, *shadeN), 0.f);
    float3 col;
    col.x = mat->spec_r * pow(specAngle, mat->exponent);
    col.y = mat->spec_g * pow(specAngle, mat->exponent);
    col.z = mat->spec_b * pow(specAngle, mat->exponent);
    
    //const Spectrum diffuse = m_matte.GetKd() * Dot(wi, Normalize(Vector(N)));
    col.x += mat->matte.r * fabs(dot(*wi, *shadeN));
    col.y += mat->matte.g * fabs(dot(*wi, *shadeN));
    col.z += mat->matte.b * fabs(dot(*wi, *shadeN));
    //col.x += mat->matte.r;
    //col.y += mat->matte.g;
    //col.z += mat->matte.b;
    //col *= INV_PI;
    
    return col;
}


//------------------------------------------------------------------------------
// Lights
//------------------------------------------------------------------------------

float3 SampleTriangleLight(__global TriangleLight *light, const float u0,
                           const float u1)
{
    
    float3 v0 = VLOAD3F(&light->v0.x);
    float3 v1 = VLOAD3F(&light->v1.x);
    float3 v2 = VLOAD3F(&light->v2.x);
    
	// UniformSampleTriangle(u0, u1, b0, b1);
	const float su1 = sqrt(u0);
	const float b0 = 1.f - su1;
	const float b1 = u1 * su1;
	const float b2 = 1.f - b0 - b1;

    return (float3) (b0*v0 + b1*v1 + b2*v2);
    
}

float TriangleArea(__global TriangleLight *light)
{

    float3 p0 = VLOAD3F(&light->v0.x);
    float3 p1 = VLOAD3F(&light->v1.x);
    float3 p2 = VLOAD3F(&light->v2.x);
    
    float3 v1 = p1 - p0;
    float3 v2 = p2 - p1;
    float3 v3 = cross(v1, v2);

    float area = 0.5f * length(v3);
    return area;
    
}

// Sample from light source
void Sample_TriangleLight_L(__global TriangleLight *l,
                            __global Ray *ray,
                            Spectrum *alpha,
                            float *pdf,
                            const float u0, const float u1,
                            const float u2, const float u3)
{
    // Do I have to flip the normal?
	float3 samplePoint = SampleTriangleLight(l, u0, u1);
    float3 sampleN = VLOAD3F(&l->normal.x);

    float3 dir = UniformSampleSphere(u2, u3);
    
    // Radiance is only emitted on one side of an area light source.
    // If the sampled direction lies on the opposite hemisphere of
    // the surface normal, we flip it.
    dir *= (dot(dir, sampleN) < 0.f) ? -1.f : 1.f;
    dir = normalize(dir);
    
    // The pdf should be a converted to a solid angle, i.e. INV_TWOPI
    *pdf = (1.f / TriangleArea(l)) * INV_TWOPI;
    
    // Get the radiance
    alpha->r = 0.f;
    alpha->g = 0.f;
    alpha->b = 0.f;
    float flip = dot(sampleN, dir);
    if (flip > 0.f) {
        alpha->r = l->gain_r;
        alpha->g = l->gain_g;
        alpha->b = l->gain_b;
    }
    
    Ray_Init4(ray, samplePoint, dir, PARAM_RAY_EPSILON, FLT_MAX);
    
}

float3 TriangleLight_Sample_L(__global TriangleLight *l,
                            const float3 *wo, const float3 *hitPoint,
                            float *pdf, Spectrum *f, Ray *shadowRay,
                            const float u0, const float u1, const float u2) {
	
    float3 samplePoint = SampleTriangleLight(l, u0, u1);
    
    float3 dir = samplePoint - (*hitPoint);
	const float distanceSquared = dot(dir, dir);
	const float distance = sqrt(distanceSquared);
	const float invDistance = 1.f / distance;
    dir *= invDistance;
    
    shadowRay->d.x = dir.x;
    shadowRay->d.y = dir.y;
    shadowRay->d.z = dir.z;
    
	float3 sampleN = VLOAD3F(&l->normal.x);
	const float sampleNdotMinusWi = -dot(sampleN, dir);
	if (sampleNdotMinusWi <= 0.f)
		*pdf = 0.f;
	else {
		*pdf = distanceSquared / (sampleNdotMinusWi * l->area);
        
		// Using 0.1 instead of 0.0 to cut down fireflies
		if (*pdf <= 0.1f)
			*pdf = 0.f;
		else {
            
            Ray_Init4_Private(shadowRay, *hitPoint, dir, PARAM_RAY_EPSILON,
                              (distance - PARAM_RAY_EPSILON));
            
            f->r = l->gain_r;
            f->g = l->gain_g;
            f->b = l->gain_b;
        }
	}
    
    return (float3)(shadowRay->d.x, shadowRay->d.y, shadowRay->d.z);
}


//------------------------------------------------------------------------------
// Init Framebuffer
//------------------------------------------------------------------------------

__kernel void InitFrameBuffer(
                              __global Pixel *frameBuffer
                              )
{
	const int gid = get_global_id(0);
	if (gid >= PARAM_IMAGE_WIDTH * PARAM_IMAGE_HEIGHT)
		return;
    
	__global Pixel *p = &frameBuffer[gid];
	p->c.r = 0.f;
	p->c.g = 0.f;
	p->c.b = 0.f;
	p->count = 0;
    
}


//------------------------------------------------------------------------------
// Initiate integration step
//------------------------------------------------------------------------------

// Sample ray directions from camera
__kernel void Init(
                   __global Path *paths,
                   __global Ray *rays
                   ) {
    
	const int gid = get_global_id(0);
	if (gid >= PARAM_PATH_COUNT)
		return;
    
	// Initialize the path
	__global Path *path = &paths[gid];
	path->throughput.x = 1.f;
	path->throughput.y = 1.f;
	path->throughput.z = 1.f;
    path->lightRadiance.x = 0.f;
	path->lightRadiance.y = 0.f;
	path->lightRadiance.z = 0.f;
	path->depth = 0;
    path->lightCount = 0;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	path->specularBounce = TRUE;
	//path->state = PATH_STATE_DIRECT_LIGHT;
    path->state = PATH_STATE_CAMERA_RAY;
	path->accumRadiance.x = 0.f;
	path->accumRadiance.y = 0.f;
	path->accumRadiance.z = 0.f;
#endif
    
    
	const uint pixelIndex = (PARAM_STARTLINE * PARAM_IMAGE_WIDTH + gid) % (PARAM_IMAGE_WIDTH * PARAM_IMAGE_HEIGHT);
	path->pixelIndex = pixelIndex;
	path->subpixelIndex = 0;
    
	// Initialize random number generator
	Seed seed;
	InitRandomGenerator(PARAM_SEED + gid, &seed);
    
	// Generate the eye ray
	GenerateRay(pixelIndex, &rays[gid], &seed);
    
	// Save the seed
	path->seed.s1 = seed.s1;
	path->seed.s2 = seed.s2;
	path->seed.s3 = seed.s3;
    
}

void UpdateFrameBuffer(__global Path *path, __global Pixel *frameBuffer,
                       float3 *radiance)
{
    // Add sample to the framebuffer
    
    const uint pixelIndex = path->pixelIndex;
    __global Pixel *pixel = &frameBuffer[pixelIndex];
    
    pixel->c.r += isnan(radiance->x) ? 0.f : radiance->x;
    pixel->c.g += isnan(radiance->y) ? 0.f : radiance->y;
    pixel->c.b += isnan(radiance->z) ? 0.f : radiance->z;
    
}

void TerminatePath(__global Path *path, __global Ray *ray,
                   __global Pixel *frameBuffer, Seed *seed, float3 *radiance
                   )
{
	// Add sample to the framebuffer
    
	const uint pixelIndex = path->pixelIndex;
	__global Pixel *pixel = &frameBuffer[pixelIndex];
    
	pixel->c.r += isnan(radiance->x) ? 0.f : radiance->x;
	pixel->c.g += isnan(radiance->y) ? 0.f : radiance->y;
	pixel->c.b += isnan(radiance->z) ? 0.f : radiance->z;
	pixel->count += 1;
    
    // Re-initialize the path
    
	uint newPixelIndex;
#if (PARAM_SAMPLE_PER_PIXEL > 1)
	const uint subpixelIndex = path->subpixelIndex;
	if (subpixelIndex >= PARAM_SAMPLE_PER_PIXEL) {
		newPixelIndex = (pixelIndex + PARAM_PATH_COUNT) % (PARAM_IMAGE_WIDTH * PARAM_IMAGE_HEIGHT);
		path->pixelIndex = newPixelIndex;
		path->subpixelIndex = 0;
	} else {
		newPixelIndex = pixelIndex;
		path->subpixelIndex = subpixelIndex + 1;
	}
#else
	newPixelIndex = (pixelIndex + PARAM_PATH_COUNT) % (PARAM_IMAGE_WIDTH * PARAM_IMAGE_HEIGHT);
	path->pixelIndex = newPixelIndex;
#endif
    
	GenerateRay(newPixelIndex, ray, seed);
    
    //path->state = PATH_STATE_DIRECT_LIGHT;
    path->state = PATH_STATE_CAMERA_RAY;
    
	path->throughput.x = 1.f;
	path->throughput.y = 1.f;
	path->throughput.z = 1.f;
    path->lightRadiance.x = 0.f;
	path->lightRadiance.y = 0.f;
	path->lightRadiance.z = 0.f;
	path->depth = 0;
    path->lightCount = 0;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	path->specularBounce = TRUE;
	path->accumRadiance.x = 0.f;
	path->accumRadiance.y = 0.f;
	path->accumRadiance.z = 0.f;
#endif
}

__kernel void Integrator(__global Path *paths,
                        __global Ray *rays,
                        __global RayHit *rayHits,
                        __global VPL *vpls, // This is a pointer to an array of VPLs
                        __global Pixel *frameBuffer,
                        __global Material *mats,
                        __global uint *meshMats,
                        __global uint *meshIDs,
                        __global Spectrum *vertColors,
                        __global Vector *vertNormals,
                        __global Triangle *triangles,
                        __global TriangleLight *triLights,
                        __global BVHAccelArrayNode *bvhTree,
                        __global Point *verts
#if defined(PARAM_HAS_TEXTUREMAPS)
                        , __global Spectrum *texMapRGBBuff
#if defined(PARAM_HAS_ALPHA_TEXTUREMAPS)
                        , __global float *texMapAlphaBuff
#endif
                        , __global TexMap *texMapDescBuff
                        , __global unsigned int *meshTexsBuff
                        , __global UV *vertUVs
#endif
 
                        )
{
	const int gid = get_global_id(0);
	
    __global Path *path = &paths[gid];
    
	__global Ray *ray = &rays[gid];
	__global RayHit *rayHit = &rayHits[gid];

    // These two lines are oh so important, without this the program does
    // not work on the GPU. The global memory has to be loaded for the data
    // to be in ray and rayHit.
    Ray rayi = *ray;
    RayHit rayhi = *rayHit;
    
    uint currentTriangleIndex = rayHit->index;
    
    __local Ray localRays[PARAM_WORK_GROUP_SIZE];
    __local RayHit rayHitsCluster[PARAM_WORK_GROUP_SIZE];
    int ltid = get_local_id(0);
    
	// Read the seed
	Seed seed;
	seed.s1 = path->seed.s1;
	seed.s2 = path->seed.s2;
	seed.s3 = path->seed.s3;
    
    if(path->state == PATH_STATE_CAMERA_RAY) {
        
        localRays[ltid] = *ray;
        Intersect(&localRays[ltid], &rayHitsCluster[ltid], verts,
                  triangles, bvhTree);
        
        *rayHit = rayHitsCluster[ltid];
        
        currentTriangleIndex = rayHitsCluster[ltid].index;
        
        path->state = PATH_STATE_DIRECT_LIGHT;
        return;
        
    }

    // Set up ray for direct lighting
    if(path->state == PATH_STATE_DIRECT_LIGHT) {
        
        if (currentTriangleIndex != 0xffffffffu) {
            // Something was hit
            const float hitPointT = rayHit->t;
            const float hitPointB1 = rayHit->b1;
            const float hitPointB2 = rayHit->b2;
            float3 rayOrig = VLOAD3F(&ray->o.x);
            float3 rayDir = VLOAD3F(&ray->d.x);
            const float3 hitPointP = rayOrig + rayHit->t * rayDir;
            
            const uint meshIndex = meshIDs[currentTriangleIndex];
            __global Material *hitPointMat = &mats[meshMats[meshIndex]];
            
            // Interpolate Color
            float3 shadeColor = Mesh_InterpolateColor(vertColors, triangles,
                                                      currentTriangleIndex,
                                                      hitPointB1, hitPointB2);
#if defined(PARAM_HAS_TEXTUREMAPS)
            // Interpolate UV coordinates
            UV uv;
            Mesh_InterpolateUV(vertUVs, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &uv);
            
            // Check it the mesh has a texture map
            unsigned int texIndex = meshTexsBuff[meshIndex];
            if (texIndex != 0xffffffffu) {
                __global TexMap *texMap = &texMapDescBuff[texIndex];
                
#if defined(PARAM_HAS_ALPHA_TEXTUREMAPS) && !defined(PARAM_DIRECT_LIGHT_SAMPLING)
                // Check if it has an alpha channel
                if (texMap->alphaOffset != 0xffffffffu) {
                    const float alpha = TexMap_GetAlpha(&texMapAlphaBuff[texMap->alphaOffset], texMap->width, texMap->height, uv.u, uv.v);
                    
                    if ((alpha == 0.0f) || ((alpha < 1.f) && (RndFloatValue(&seed) > alpha))) {
                        // Continue to trace the ray
                        matType = MAT_NULL;
                    }
                }
#endif
                
                Spectrum texColor;
                TexMap_GetColor(&texMapRGBBuff[texMap->rgbOffset], texMap->width, texMap->height, uv.u, uv.v, &texColor);
                
                shadeColor.x *= texColor.r;
                shadeColor.y *= texColor.g;
                shadeColor.z *= texColor.b;
            }
#endif
            
            // Interpolate the normal
            float3 shadeN = Mesh_InterpolateNormal(vertNormals, triangles, currentTriangleIndex,
                                                   hitPointB1, hitPointB2);
            shadeN *= (dot(rayDir, shadeN) > 0.f) ? -1.f : 1.f;
            
            float directLightPdf;
            switch (hitPointMat->type) {
                case MAT_MATTE:
                    directLightPdf = 1.f;
                    break;
                case MAT_BLINNPHONG: {
                    directLightPdf = 1.f;
                    break;
                }
                default:
                    directLightPdf = 0.f;
                    break;
            }
            
            if (directLightPdf > 0.f) {
                // Select a light source to sample
                const uint lightIndex = min((uint)floor(PARAM_DL_LIGHT_COUNT * RndFloatValue(&seed)),
                                            (uint)(PARAM_DL_LIGHT_COUNT - 1));
                __global TriangleLight *l = &triLights[lightIndex];
                
                // Setup the shadow ray
                float3 wo = -rayDir;
                
                float3 Le;
                float lightPdf;
                Ray shadowRay;
                float3 dir = TriangleLight_Sample_L(l, &wo, &hitPointP, &lightPdf, &Le,
                                                    &shadowRay, RndFloatValue(&seed),
                                                    RndFloatValue(&seed), RndFloatValue(&seed));
                
                const float dp = dot(shadeN, dir);
                const float matPdf = (dp <= 0.f) ? 0.f : 1.f;
                
                const float pdf = lightPdf * matPdf * directLightPdf;
                if (pdf > 0.f) {
                    
                    const float k = dp * PARAM_DL_LIGHT_COUNT / (pdf);
                    
                    if(hitPointMat->type == MAT_MATTE) {
                        
                        float3 col = Matte_f(&hitPointMat->param, &wo, &dir, &shadeN);
                        col *= k;
                        path->lightRadiance = col * Le;
                        
                    }else if(hitPointMat->type == MAT_BLINNPHONG) {
                        
                        float3 col = BlinnPhong_f(&hitPointMat->param, &wo, &dir, &shadeN);
                        col *= k;
                        path->lightRadiance = col * Le;
                    }
                    
                }
                
                localRays[ltid] = shadowRay;
                Intersect(&localRays[ltid], &rayHitsCluster[ltid], verts,
                          triangles, bvhTree);
                
                // Add direct radiance
                if (rayHitsCluster[ltid].index == 0xffffffffu) {
                    // Nothing was hit, the light is visible
                    path->lightRadiance *= shadeColor;
                    float3 radiance = path->lightRadiance;
                    UpdateFrameBuffer(path, frameBuffer, &radiance);
                }
                
            }
            
            path->state = PATH_STATE_VPL_LIGHT;
            
          
        }
        else {
            
            float3 radiance;
            
            radiance = (float3)(0.0f, 0.0f, 0.0f);
            
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
            radiance += path->accumRadiance;
#endif
            TerminatePath(path, ray, frameBuffer, &seed, &radiance);
        }
        
        // Save the seed
        path->seed.s1 = seed.s1;
        path->seed.s2 = seed.s2;
        path->seed.s3 = seed.s3;
        
        return;
        
    }
    
    if (path->state == PATH_STATE_VPL_LIGHT) {
    
        const float hitPointT = rayHit->t;
        const float hitPointB1 = rayHit->b1;
        const float hitPointB2 = rayHit->b2;
        
        float3 rayOrig = VLOAD3F(&ray->o.x);
        float3 rayDir = VLOAD3F(&ray->d.x);
        
        const float3 hitPointP = rayOrig + rayHit->t * rayDir;
        
        int gpuLoopLimit = path->lightCount + 4;
        
        // Loop here
#if defined DEVICE_CPU
        for(; path->lightCount < PARAM_NLIGHTS; path->lightCount++) {
#else
        for(; path->lightCount < PARAM_NLIGHTS && path->lightCount < gpuLoopLimit; path->lightCount++) {
#endif
                
            __global VPL *vpl = &vpls[path->lightCount];
            
            float3 vplHitPoint = VLOAD3F(&vpl->hitPoint.x);
            
            float3 d = hitPointP - vplHitPoint;
            const float d2 = length(d*d);
            
            float3 shadeN = Mesh_InterpolateNormal(vertNormals, triangles, currentTriangleIndex,
                                                   hitPointB1, hitPointB2);
            shadeN *= (dot(rayDir, shadeN) > 0.f) ? -1.f : 1.f;
            
            float3 vplN = VLOAD3F(&vpl->n.x);
            float3 wi = normalize(vplHitPoint - hitPointP);
            float G = max(0.f, dot(wi,shadeN)) * fabs(dot(wi, vplN)) / d2;
            G = min(G, .66f);
            
            const uint meshIndex = meshIDs[currentTriangleIndex];
            __global Material *hitPointMat = &mats[meshMats[meshIndex]];
            float3 wo = -rayDir;
            
            float3 f = (float3)(0.0f, 0.0f, 0.0f);
            if(hitPointMat->type == MAT_MATTE || hitPointMat->type == MAT_AREALIGHT ) {
                f = Matte_f(&hitPointMat->param, &wo, &wi, &shadeN);
            }else if(hitPointMat->type == MAT_BLINNPHONG) {
                f = BlinnPhong_f(&hitPointMat->param, &wo, &wi, &shadeN);
            }
            
            if (G == 0.f || (f.x == 0.f && f.y == 0.f && f.z == 0.f))
                continue;
            
            float3 radiance = (float3)(0.0f, 0.0f, 0.0f);
            radiance.x = f.x * G * vpl->contrib.r / PARAM_NLIGHTS;
            radiance.y = f.y * G * vpl->contrib.g / PARAM_NLIGHTS;
            radiance.z = f.z * G * vpl->contrib.b / PARAM_NLIGHTS;
            
            // Russian roulette
            if(Y(radiance) < 0.0001f) {
                float continueProbability = .1f;
                if(RndFloatValue(&seed) > continueProbability) {
                    continue;
                }
                radiance /= continueProbability;
            }
            
            // Shadow ray test
            float3 dir = vplHitPoint - hitPointP;
            const float distanceSquared = dot(dir, dir);
            const float distance = sqrt(distanceSquared);
            const float invDistance = 1.f / distance;
            dir *= invDistance;
            Ray_Init4_Local(&localRays[ltid], hitPointP, dir, PARAM_RAY_EPSILON,
                            (distance - PARAM_RAY_EPSILON));
            Intersect(&localRays[ltid], &rayHitsCluster[ltid], verts,
                      triangles, bvhTree);
            
            uint hitVPL = rayHitsCluster[ltid].index;
        
            // VPL is visible
            if(hitVPL == 0xffffffffu) {
                path->accumRadiance += radiance;
            }
#if defined DEVICE_CPU
        } // end for loop CPU
        
        float3 shadeColor = (float3)(1.f, 1.f, 1.f);
#if defined(PARAM_HAS_TEXTUREMAPS)
            
        const uint meshIndex = meshIDs[currentTriangleIndex];
        
        // Interpolate UV coordinates
        UV uv;
        Mesh_InterpolateUV(vertUVs, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &uv);
        
        // Check it the mesh has a texture map
        unsigned int texIndex = meshTexsBuff[meshIndex];
        if (texIndex != 0xffffffffu) {
            __global TexMap *texMap = &texMapDescBuff[texIndex];
            
            Spectrum texColor;
            TexMap_GetColor(&texMapRGBBuff[texMap->rgbOffset], texMap->width, texMap->height, uv.u, uv.v, &texColor);
            
            shadeColor.x *= texColor.r;
            shadeColor.y *= texColor.g;
            shadeColor.z *= texColor.b;
        }
#endif
    
        float3 radiance;
        
        radiance = (float3)(0.0f, 0.0f, 0.0f);
        
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
        radiance = path->accumRadiance;
        radiance *= shadeColor;
#endif
        TerminatePath(path, ray, frameBuffer, &seed, &radiance);
        
#else
        } // end for loop GPU
        
        if(path->lightCount == PARAM_NLIGHTS) {
            
            float3 shadeColor = (float3)(1.f, 1.f, 1.f);
#if defined(PARAM_HAS_TEXTUREMAPS)
            
            const uint meshIndex = meshIDs[currentTriangleIndex];
            
            // Interpolate UV coordinates
            UV uv;
            Mesh_InterpolateUV(vertUVs, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &uv);
            
            // Check it the mesh has a texture map
            unsigned int texIndex = meshTexsBuff[meshIndex];
            if (texIndex != 0xffffffffu) {
                __global TexMap *texMap = &texMapDescBuff[texIndex];
                
                Spectrum texColor;
                TexMap_GetColor(&texMapRGBBuff[texMap->rgbOffset], texMap->width, texMap->height, uv.u, uv.v, &texColor);
                
                shadeColor.x *= texColor.r;
                shadeColor.y *= texColor.g;
                shadeColor.z *= texColor.b;
            }
#endif
            
            float3 radiance;
            
            radiance = (float3)(0.0f, 0.0f, 0.0f);
            
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
            radiance = path->accumRadiance;
            radiance *= shadeColor;
#endif
            TerminatePath(path, ray, frameBuffer, &seed, &radiance);
        }
        else {
            //path->accumRadiance = totalRadiance;
            
            path->state = PATH_STATE_VPL_LIGHT;

        }
#endif
            
    }
    
    // Save the seed
    path->seed.s1 = seed.s1;
    path->seed.s2 = seed.s2;
    path->seed.s3 = seed.s3;
    
}


