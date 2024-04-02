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

typedef struct {
    
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
    Ray pathRay;
    RayHit pathHit;
    float3 lightRadiance;
    float3 accumRadiance;
#endif
    
    float3 throughput;
    Seed seed;
    
    uint depth, pixelIndex, subpixelIndex;
    uint lightCount;
    int specularBounce;
    int state;
    int clusterIndex;
    
} Path;

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
        BlinnPhongParam blinnPhong;
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

float3 VLOAD3F(const __global float *p) {
    return (float3)(p[0], p[1], p[2]);
}

void LVSTORE3F(const float3 v, __local float *p) {
    p[0] = v.x;
    p[1] = v.y;
    p[2] = v.z;
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

void Ray_Init4(__global Ray *ray, const float3 orig, const float3 dir,
               const float mint, const float maxt) {
    VSTORE3F(orig, &ray->o.x);
    VSTORE3F(dir, &ray->d.x);
    ray->mint = mint;
    ray->maxt = maxt;
}

//------------------------------------------------------------------------------

float Dot(const Vector *v0, const Vector *v1) {
	return v0->x * v1->x + v0->y * v1->y + v0->z * v1->z;
}

void Normalize(Vector *v) {
	const float il = 1.f / sqrt(Dot(v, v));
    
	v->x *= il;
	v->y *= il;
	v->z *= il;
}

void Cross(Vector *v3, const Vector *v1, const Vector *v2) {
	v3->x = (v1->y * v2->z) - (v1->z * v2->y);
	v3->y = (v1->z * v2->x) - (v1->x * v2->z),
	v3->z = (v1->x * v2->y) - (v1->y * v2->x);
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

void CosineSampleHemisphere(Vector *ret, const float u1, const float u2) {
	ConcentricSampleDisk(u1, u2, &ret->x, &ret->y);
	ret->z = sqrt(max(0.f, 1.f - ret->x * ret->x - ret->y * ret->y));
}

void CoordinateSystem(const Vector *v1, Vector *v2, Vector *v3) {
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
    
	Cross(v3, v1, v2);
}

void SphericalDirection(Vector *vec, float sintheta, float costheta, float phi) {
    vec->x = sintheta * cos(phi);
    vec->y = sintheta * sin(phi);
    vec->z = costheta;
}

bool SameHemisphere(const Vector *w, const Vector *wp) {
    return w->z * wp->z > 0.f;
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
    
	Point Pras;
	Pras.x = screenX;
	Pras.y = PARAM_IMAGE_HEIGHT - screenY - 1.f;
	Pras.z = 0;
    
	Point orig;
	// RasterToCamera(Pras, &orig);
	const float iw = 1.f / (PARAM_RASTER2CAMERA_30 * Pras.x + PARAM_RASTER2CAMERA_31 * Pras.y + PARAM_RASTER2CAMERA_32 * Pras.z + PARAM_RASTER2CAMERA_33);
	orig.x = (PARAM_RASTER2CAMERA_00 * Pras.x + PARAM_RASTER2CAMERA_01 * Pras.y + PARAM_RASTER2CAMERA_02 * Pras.z + PARAM_RASTER2CAMERA_03) * iw;
	orig.y = (PARAM_RASTER2CAMERA_10 * Pras.x + PARAM_RASTER2CAMERA_11 * Pras.y + PARAM_RASTER2CAMERA_12 * Pras.z + PARAM_RASTER2CAMERA_13) * iw;
	orig.z = (PARAM_RASTER2CAMERA_20 * Pras.x + PARAM_RASTER2CAMERA_21 * Pras.y + PARAM_RASTER2CAMERA_22 * Pras.z + PARAM_RASTER2CAMERA_23) * iw;
    
	Vector dir;
	dir.x = orig.x;
	dir.y = orig.y;
	dir.z = orig.z;
    
#if defined(PARAM_CAMERA_HAS_DOF)
	// Sample point on lens
	float lensU, lensV;
	ConcentricSampleDisk(RndFloatValue(seed), RndFloatValue(seed), &lensU, &lensV);
	lensU *= PARAM_CAMERA_LENS_RADIUS;
	lensV *= PARAM_CAMERA_LENS_RADIUS;
    
	// Compute point on plane of focus
	const float ft = (PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / dir.z;
	Point Pfocus;
	Pfocus.x = orig.x + dir.x * ft;
	Pfocus.y = orig.y + dir.y * ft;
	Pfocus.z = orig.z + dir.z * ft;
    
	// Update ray for effect of lens
	orig.x += lensU * ((PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / PARAM_CAMERA_FOCAL_DISTANCE);
	orig.y += lensV * ((PARAM_CAMERA_FOCAL_DISTANCE - PARAM_CLIP_HITHER) / PARAM_CAMERA_FOCAL_DISTANCE);
    
	dir.x = Pfocus.x - orig.x;
	dir.y = Pfocus.y - orig.y;
	dir.z = Pfocus.z - orig.z;
#endif
    
	Normalize(&dir);
    
	// CameraToWorld(*ray, ray);
	Point torig;
	const float iw2 = 1.f / (PARAM_CAMERA2WORLD_30 * orig.x + PARAM_CAMERA2WORLD_31 * orig.y + PARAM_CAMERA2WORLD_32 * orig.z + PARAM_CAMERA2WORLD_33);
	torig.x = (PARAM_CAMERA2WORLD_00 * orig.x + PARAM_CAMERA2WORLD_01 * orig.y + PARAM_CAMERA2WORLD_02 * orig.z + PARAM_CAMERA2WORLD_03) * iw2;
	torig.y = (PARAM_CAMERA2WORLD_10 * orig.x + PARAM_CAMERA2WORLD_11 * orig.y + PARAM_CAMERA2WORLD_12 * orig.z + PARAM_CAMERA2WORLD_13) * iw2;
	torig.z = (PARAM_CAMERA2WORLD_20 * orig.x + PARAM_CAMERA2WORLD_21 * orig.y + PARAM_CAMERA2WORLD_22 * orig.z + PARAM_CAMERA2WORLD_23) * iw2;
    
	Vector tdir;
	tdir.x = PARAM_CAMERA2WORLD_00 * dir.x + PARAM_CAMERA2WORLD_01 * dir.y + PARAM_CAMERA2WORLD_02 * dir.z;
	tdir.y = PARAM_CAMERA2WORLD_10 * dir.x + PARAM_CAMERA2WORLD_11 * dir.y + PARAM_CAMERA2WORLD_12 * dir.z;
	tdir.z = PARAM_CAMERA2WORLD_20 * dir.x + PARAM_CAMERA2WORLD_21 * dir.y + PARAM_CAMERA2WORLD_22 * dir.z;
    
	ray->o = torig;
	ray->d = tdir;
	ray->mint = PARAM_RAY_EPSILON;
	ray->maxt = (PARAM_CLIP_YON - PARAM_CLIP_HITHER) / dir.z;
}

//------------------------------------------------------------------------------

__kernel void Init(
                   __global Path *paths,
                   __global Ray *rays
                   ) {
    
	const int gid = get_global_id(0);
	if (gid >= PARAM_PATH_COUNT)
		return;
    
	// Initialize the path
	__global Path *path = &paths[gid];
    path->throughput = (float3)(1.0f, 1.0f, 1.0f);
	path->depth = 0;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	path->specularBounce = TRUE;
	path->state = PATH_STATE_NEXT_VERTEX;
	path->accumRadiance = (float3)(0.0f, 0.0f, 0.0f);
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

//------------------------------------------------------------------------------

__kernel void InitFrameBuffer(
                              __global Pixel *frameBuffer
                              ) {
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

float SphericalTheta(const Vector *v) {
	return acos(clamp(v->z, -1.f, 1.f));
}

float SphericalPhi(const Vector *v) {
	float p = atan2(v->y, v->x);
	return (p < 0.f) ? p + 2.f * M_PI : p;
}

void Mesh_InterpolateColor(__global Spectrum *colors, __global Triangle *triangles,
                           const uint triIndex, const float b1, const float b2, Spectrum *C) {
	__global Triangle *tri = &triangles[triIndex];
    
	const float b0 = 1.f - b1 - b2;
	C->r = b0 * colors[tri->v0].r + b1 * colors[tri->v1].r + b2 * colors[tri->v2].r;
	C->g = b0 * colors[tri->v0].g + b1 * colors[tri->v1].g + b2 * colors[tri->v2].g;
	C->b = b0 * colors[tri->v0].b + b1 * colors[tri->v1].b + b2 * colors[tri->v2].b;
}

void Mesh_InterpolateNormal(__global Vector *normals, __global Triangle *triangles,
                            const uint triIndex, const float b1, const float b2, Vector *N) {
	__global Triangle *tri = &triangles[triIndex];
    
	const float b0 = 1.f - b1 - b2;
	N->x = b0 * normals[tri->v0].x + b1 * normals[tri->v1].x + b2 * normals[tri->v2].x;
	N->y = b0 * normals[tri->v0].y + b1 * normals[tri->v1].y + b2 * normals[tri->v2].y;
	N->z = b0 * normals[tri->v0].z + b1 * normals[tri->v1].z + b2 * normals[tri->v2].z;
    
	Normalize(N);
}

void Mesh_InterpolateUV(__global UV *uvs, __global Triangle *triangles,
                        const uint triIndex, const float b1, const float b2, UV *uv) {
    __global Triangle *tri = &triangles[triIndex];
    
    const float b0 = 1.f - b1 - b2;
    uv->u = b0 * uvs[tri->v0].u + b1 * uvs[tri->v1].u + b2 * uvs[tri->v2].u;
    uv->v = b0 * uvs[tri->v0].v + b1 * uvs[tri->v1].v + b2 * uvs[tri->v2].v;
}

//------------------------------------------------------------------------------
// Materials
//------------------------------------------------------------------------------

void Matte_Sample_f(__global MatteParam *mat, const Vector *wo, Vector *wi,
                    float *pdf, Spectrum *f, const Vector *shadeN,
                    const float u0, const float u1
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
                    , __global int *specularBounce
#endif
                    ) {
	Vector dir;
	CosineSampleHemisphere(&dir, u0, u1);
	*pdf = dir.z * INV_PI;
    
    // Local to world
	Vector v1, v2;
	CoordinateSystem(shadeN, &v1, &v2);
	wi->x = v1.x * dir.x + v2.x * dir.y + shadeN->x * dir.z;
	wi->y = v1.y * dir.x + v2.y * dir.y + shadeN->y * dir.z;
	wi->z = v1.z * dir.x + v2.z * dir.y + shadeN->z * dir.z;
    
	const float dp = Dot(shadeN, wi);
	// Using 0.0001 instead of 0.0 to cut down fireflies
	if (dp <= 0.0001f)
		*pdf = 0.f;
	else {
		*pdf /=  dp;
        
		f->r = mat->r * INV_PI;
		f->g = mat->g * INV_PI;
		f->b = mat->b * INV_PI;
	}
    
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	*specularBounce = FALSE;
#endif
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

void BlinnPhong_Sample_f(__global BlinnPhongParam *mat, const Vector *wo,
                         Vector *wi, float *pdf, Spectrum *f,
                         const Vector *shadeN, const float u0, const float u1)
{
    // Compute sampled half-angle vector $\wh$ for Blinn distribution
    float costheta = pow(u0, 1.f / (mat->exponent+1.f));
    float sintheta = sqrt(max(0.f, 1.f - costheta*costheta));
    float phi = u1 * 2.f * M_PI;
    Vector wh;
    SphericalDirection(&wh, sintheta, costheta, phi);
    
    // Local to world
    Vector v1, v2;
    CoordinateSystem(shadeN, &v1, &v2);
    wh.x = v1.x * wh.x + v2.x * wh.y + shadeN->x * wh.z;
    wh.y = v1.y * wh.x + v2.y * wh.y + shadeN->y * wh.z;
    wh.z = v1.z * wh.x + v2.z * wh.y + shadeN->z * wh.z;
    /*
    if (!SameHemisphere(wo, &wh)) {
        wh.x = -wh.x;
        wh.y = -wh.y;
        wh.z = -wh.z;
    }
    */
    // Compute incident direction by reflecting about $\wh$
    wi->x = -wo->x + 2.f * Dot(wo, &wh) * wh.x;
    wi->y = -wo->y + 2.f * Dot(wo, &wh) * wh.y;
    wi->z = -wo->z + 2.f * Dot(wo, &wh) * wh.z;
    /*
    if (!SameHemisphere(wo, wi)) {
        *pdf = 0.f;
        f->r = 0.f;
        f->g = 0.f;
        f->b = 0.f;
        return;
    }
    */
    // Compute PDF for $\wi$ from Blinn distribution
    float blinn_pdf = ((mat->exponent + 1.f) * pow(costheta, mat->exponent)) /
    (2.f * M_PI * 4.f * Dot(wo, &wh));
    if (Dot(wo, &wh) <= 0.f) blinn_pdf = 0.f;
    *pdf = blinn_pdf;
    *pdf = 1.f;
    /*
    float specAngle = max(Dot(&wh, shadeN), 0.f);
    f->r = mat->spec_r * pow(specAngle, mat->exponent);
    f->g = mat->spec_g * pow(specAngle, mat->exponent);
    f->b = mat->spec_b * pow(specAngle, mat->exponent);
    */
    f->r += mat->matte.r * fabs(Dot(wi, shadeN));
    f->g += mat->matte.g * fabs(Dot(wi, shadeN));
    f->b += mat->matte.b * fabs(Dot(wi, shadeN));
    
    //*pdf += SameHemisphere(wo, wi) ? fabs(wi->z) * INV_PI : 0.f;
    //*pdf /= 2;
    
}

float3 BlinnPhong_f(__global BlinnPhongParam *mat, const float3 *wo,
                    const float3 *wi, const float3 *shadeN)
{
    float cosThetaO = fabs(wo->z);
    float cosThetaI = fabs(wi->z);
    if (cosThetaI == 0.f || cosThetaO == 0.f) return (float3)(0.0f, 0.0f, 0.0f);
    float3 wh = *wi + *wo;
    if (wh.x == 0. && wh.y == 0. && wh.z == 0.) return (float3)(0.0f, 0.0f, 0.0f);
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

void AreaLight_Le(__global AreaLightParam *mat, const Vector *wo, const Vector *lightN, Spectrum *Le) {
	if (Dot(lightN, wo) > 0.f) {
        Le->r = mat->gain_r;
        Le->g = mat->gain_g;
        Le->b = mat->gain_b;
    }
}

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

float3 TriangleLight_Sample_L(__global TriangleLight *l,
                              const float3 *wo, const float3 *hitPoint,
                              float *pdf, float3 *f, Ray *shadowRay,
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
            
            f->x = l->gain_r;
            f->y = l->gain_g;
            f->z = l->gain_b;
        }
    }
    
    return (float3)(shadowRay->d.x, shadowRay->d.y, shadowRay->d.z);
}

//------------------------------------------------------------------------------

void TerminatePath(__global Path *path, __global Ray *ray, __global Pixel *frameBuffer, Seed *seed, Spectrum *radiance
                   ) {
	// Add sample to the framebuffer
    
	const uint pixelIndex = path->pixelIndex;
	__global Pixel *pixel = &frameBuffer[pixelIndex];
    
	pixel->c.r += isnan(radiance->r) ? 0.f : radiance->r;
	pixel->c.g += isnan(radiance->g) ? 0.f : radiance->g;
	pixel->c.b += isnan(radiance->b) ? 0.f : radiance->b;
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
    
	path->throughput = (float3)(1.0f, 1.0f, 1.0f);
	path->depth = 0;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	path->specularBounce = TRUE;
	path->state = PATH_STATE_NEXT_VERTEX;
	path->accumRadiance = (float3)(0.0f, 0.0f, 0.0f);
#endif
}

#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
__kernel void AdvancePaths_Step1(
                                 __global Path *paths,
                                 __global Ray *rays,
                                 __global RayHit *rayHits,
                                 __global Material *mats,
                                 __global uint *meshMats,
                                 __global uint *meshIDs,
                                 __global Spectrum *vertColors,
                                 __global Vector *vertNormals,
                                 __global Triangle *triangles,
                                 __global TriangleLight *triLights
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
	uint currentTriangleIndex = rayHit->index;
    
	// Read the seed
	Seed seed;
	seed.s1 = path->seed.s1;
	seed.s2 = path->seed.s2;
	seed.s3 = path->seed.s3;
    
    //printf("path state: %u seed: %u", path->state, path->seed);
    
	if (path->state == PATH_STATE_SAMPLE_LIGHT) {
		if (currentTriangleIndex == 0xffffffffu) {
			// Nothing was hit, the light is visible
			path->accumRadiance += path->lightRadiance;
		}
#if defined(PARAM_HAS_ALPHA_TEXTUREMAPS)
        else {
            const uint meshIndex = meshIDs[currentTriangleIndex];
            
            // Check if is a mesh with alpha texture map applied
            const float hitPointB1 = rayHit->b1;
            const float hitPointB2 = rayHit->b2;
            
            // Check it the mesh has a texture map
            unsigned int texIndex = meshTexsBuff[meshIndex];
            if (texIndex != 0xffffffffu) {
                __global TexMap *texMap = &texMapDescBuff[texIndex];
                
                // Check if it has an alpha channel
                if (texMap->alphaOffset != 0xffffffffu) {
                    // Interpolate UV coordinates
                    UV uv;
                    Mesh_InterpolateUV(vertUVs, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &uv);
                    
                    const float alpha = TexMap_GetAlpha(&texMapAlphaBuff[texMap->alphaOffset], texMap->width, texMap->height, uv.u, uv.v);
                    
                    if ((alpha == 0.0f) || ((alpha < 1.f) && (RndFloatValue(&seed) > alpha))) {
                        // Continue to trace the ray
                        const float hitPointT = rayHit->t;
                        ray->o.x = ray->o.x + ray->d.x * hitPointT;
                        ray->o.y = ray->o.y + ray->d.y * hitPointT;
                        ray->o.z = ray->o.z + ray->d.z * hitPointT;
                        ray->maxt -= hitPointT;
                        
                        // Save the seed
                        path->seed.s1 = seed.s1;
                        path->seed.s2 = seed.s2;
                        path->seed.s3 = seed.s3;
                        
                        return;
                    }
                }
            }
            
        } // end define alpha maps
#endif

        
		// Restore the path RayHit
		*rayHit = path->pathHit;
        
		// Restore the path Ray
        *ray = path->pathRay;
        
        path->state = PATH_STATE_NEXT_VERTEX;
        
        return;
	}
    
    // state is PATH_STATE_NEXT_VERTEX or PATH_STATE_CONTINUE_NEXT_VERTEX
    
	// If we are in PATH_STATE_CONTINUE_NEXT_VERTEX state
	path->state = PATH_STATE_NEXT_VERTEX;
    
    if (currentTriangleIndex != 0xffffffffu) {
        // Something was hit
        
		const uint meshIndex = meshIDs[currentTriangleIndex];
		__global Material *hitPointMat = &mats[meshMats[meshIndex]];
        
        Vector rayDir = ray->d;
        
        const float hitPointT = rayHit->t;
        const float hitPointB1 = rayHit->b1;
        const float hitPointB2 = rayHit->b2;
        
        Point hitPoint;
        hitPoint.x = ray->o.x + rayDir.x * hitPointT;
        hitPoint.y = ray->o.y + rayDir.y * hitPointT;
        hitPoint.z = ray->o.z + rayDir.z * hitPointT;
        
		// Interpolate Color
		Spectrum shadeColor;
		Mesh_InterpolateColor(vertColors, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &shadeColor);
        
#if defined(PARAM_HAS_TEXTUREMAPS)
        // Interpolate UV coordinates
        UV uv;
        Mesh_InterpolateUV(vertUVs, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &uv);
        
        // Check it the mesh has a texture map
        unsigned int texIndex = meshTexsBuff[meshIndex];
        if (texIndex != 0xffffffffu) {
            __global TexMap *texMap = &texMapDescBuff[texIndex];
            
#if defined(PARAM_HAS_ALPHA_TEXTUREMAPS)
            // Check if it has an alpha channel
            if (texMap->alphaOffset != 0xffffffffu) {
                const float alpha = TexMap_GetAlpha(&texMapAlphaBuff[texMap->alphaOffset], texMap->width, texMap->height, uv.u, uv.v);
                
                if ((alpha == 0.0f) || ((alpha < 1.f) && (RndFloatValue(&seed) > alpha))) {
                    // Continue to trace the ray
                    ray->o = hitPoint;
                    ray->maxt -= hitPointT;
                    
                    path->state = PATH_STATE_CONTINUE_NEXT_VERTEX;
                    // Save the seed
                    path->seed.s1 = seed.s1;
                    path->seed.s2 = seed.s2;
                    path->seed.s3 = seed.s3;
                    
                    return;
                }
            }
#endif
            
            Spectrum texColor;
            TexMap_GetColor(&texMapRGBBuff[texMap->rgbOffset], texMap->width, texMap->height, uv.u, uv.v, &texColor);
            
            shadeColor.r *= texColor.r;
            shadeColor.g *= texColor.g;
            shadeColor.b *= texColor.b;
        }
#endif
        
		// Interpolate the normal
        Vector N;
		Mesh_InterpolateNormal(vertNormals, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &N);
        
		// Flip the normal if required
        Vector shadeN;
		const float nFlip = (Dot(&rayDir, &N) > 0.f) ? -1.f : 1.f;
		shadeN.x = nFlip * N.x;
		shadeN.y = nFlip * N.y;
		shadeN.z = nFlip * N.z;
        
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
            const uint lightIndex = min((uint)floor(PARAM_DL_LIGHT_COUNT * RndFloatValue(&seed)), (uint)(PARAM_DL_LIGHT_COUNT - 1));
            __global TriangleLight *l = &triLights[lightIndex];
            
            // Setup the shadow ray
            Vector wo;
            wo.x = -rayDir.x;
            wo.y = -rayDir.y;
            wo.z = -rayDir.z;
            float3 Le;
            float lightPdf;
            Ray shadowRay;
            float3 dir = TriangleLight_Sample_L(l, &wo, &hitPoint, &lightPdf, &Le, &shadowRay,
                                   RndFloatValue(&seed), RndFloatValue(&seed), RndFloatValue(&seed));
            
            const float dp = Dot(&shadeN, &shadowRay.d);
            const float matPdf = (dp <= 0.f) ? 0.f : 1.f;
            
            const float pdf = lightPdf * matPdf * directLightPdf;
            if (pdf > 0.f) {
                float3 throughput = path->throughput;
                throughput.x *= shadeColor.r;
                throughput.y *= shadeColor.g;
                throughput.z *= shadeColor.b;
                
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
                path->lightRadiance *= throughput;
                /*
                const float k = dp * PARAM_DL_LIGHT_COUNT / (pdf * M_PI);
                // NOTE: I assume all matte mixed material have a MatteParam as first field
                path->lightRadiance.x = throughput.x * hitPointMat->param.matte.r * k * Le.r;
                path->lightRadiance.y = throughput.y * hitPointMat->param.matte.g * k * Le.g;
                path->lightRadiance.z = throughput.z * hitPointMat->param.matte.b * k * Le.b;
                
                if(hitPointMat->type == MAT_BLINNPHONG) {
                    
                    float3 wh;
                    wh.x = shadowRay.d.x + wo.x;
                    wh.y = shadowRay.d.y + wo.y;
                    wh.z = shadowRay.d.z + wo.z;
                    wh = normalize(wh);
                    
                    float3 shadingNormal;
                    shadingNormal.x = shadeN.x;
                    shadingNormal.y = shadeN.y;
                    shadingNormal.z = shadeN.z;
                    
                    float specAngle = max(dot(wh, shadingNormal), 0.f);
                    float3 col;
                    path->lightRadiance.x += hitPointMat->param.blinnPhong.spec_r * pow(specAngle, hitPointMat->param.blinnPhong.exponent) * k * Le.r;
                    path->lightRadiance.y += hitPointMat->param.blinnPhong.spec_g * pow(specAngle, hitPointMat->param.blinnPhong.exponent) * k * Le.g;
                    path->lightRadiance.z += hitPointMat->param.blinnPhong.spec_b * pow(specAngle, hitPointMat->param.blinnPhong.exponent) * k * Le.b;
                 
                }*/
                
                // Save current ray hit information
                path->pathHit.t = hitPointT;
                path->pathHit.b1 = hitPointB1;
                path->pathHit.b2 = hitPointB2;
                path->pathHit.index = currentTriangleIndex;
                
                // Save the current Ray
                path->pathRay = *ray;
                
                *ray = shadowRay;
                
                path->state = PATH_STATE_SAMPLE_LIGHT;
            }
            
            // Save the seed
            path->seed.s1 = seed.s1;
            path->seed.s2 = seed.s2;
            path->seed.s3 = seed.s3;
        }
    }
}
#endif

__kernel void AdvancePaths_Step2(
                                 __global Path *paths,
                                 __global Ray *rays,
                                 __global RayHit *rayHits,
                                 __global Pixel *frameBuffer,
                                 __global Material *mats,
                                 __global uint *meshMats,
                                 __global uint *meshIDs,
                                 __global Spectrum *vertColors,
                                 __global Vector *vertNormals,
                                 __global Triangle *triangles
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
                                 , __global TriangleLight *triLights
#endif
#if defined(PARAM_HAS_TEXTUREMAPS)
                                 , __global Spectrum *texMapRGBBuff
#if defined(PARAM_HAS_ALPHA_TEXTUREMAPS)
                                 , __global float *texMapAlphaBuff
#endif
                                 , __global TexMap *texMapDescBuff
                                 , __global unsigned int *meshTexsBuff
                                 , __global UV *vertUVs
#endif
                                 ) {
	const int gid = get_global_id(0);
	__global Path *path = &paths[gid];
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
	int pathState = path->state;
	if ((pathState == PATH_STATE_SAMPLE_LIGHT) || (pathState == PATH_STATE_CONTINUE_NEXT_VERTEX)) {
        // Handled by Step1 kernel
        return;
    }
#endif
    
	// Read the seed
	Seed seed;
	seed.s1 = path->seed.s1;
	seed.s2 = path->seed.s2;
	seed.s3 = path->seed.s3;
    
    __global Ray *ray = &rays[gid];
	__global RayHit *rayHit = &rayHits[gid];
	uint currentTriangleIndex = rayHit->index;
        
	const float hitPointT = rayHit->t;
    const float hitPointB1 = rayHit->b1;
    const float hitPointB2 = rayHit->b2;
    
    Vector rayDir = ray->d;
    
	Point hitPoint;
    hitPoint.x = ray->o.x + rayDir.x * hitPointT;
    hitPoint.y = ray->o.y + rayDir.y * hitPointT;
    hitPoint.z = ray->o.z + rayDir.z * hitPointT;
    
	float3 throughput = path->throughput;
    
    if (currentTriangleIndex != 0xffffffffu) {
		// Something was hit
        
		const uint meshIndex = meshIDs[currentTriangleIndex];
		__global Material *hitPointMat = &mats[meshMats[meshIndex]];
		uint matType = hitPointMat->type;
        
		// Interpolate Color
        Spectrum shadeColor;
		Mesh_InterpolateColor(vertColors, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &shadeColor);
        
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
            
            shadeColor.r *= texColor.r;
            shadeColor.g *= texColor.g;
            shadeColor.b *= texColor.b;
        }
#endif
        
		throughput.x *= shadeColor.r;
		throughput.y *= shadeColor.g;
		throughput.z *= shadeColor.b;
        
		// Interpolate the normal
        Vector N;
		Mesh_InterpolateNormal(vertNormals, triangles, currentTriangleIndex, hitPointB1, hitPointB2, &N);
        
		// Flip the normal if required
        Vector shadeN;
		const float nFlip = (Dot(&rayDir, &N) > 0.f) ? -1.f : 1.f;
		shadeN.x = nFlip * N.x;
		shadeN.y = nFlip * N.y;
		shadeN.z = nFlip * N.z;
        
		const float u0 = RndFloatValue(&seed);
		const float u1 = RndFloatValue(&seed);
		const float u2 = RndFloatValue(&seed);
        
		Vector wo;
		wo.x = -rayDir.x;
		wo.y = -rayDir.y;
		wo.z = -rayDir.z;
        
		Vector wi;
		float pdf;
		Spectrum f;
        
		switch (matType) {
                
#if defined(PARAM_ENABLE_MAT_MATTE)
			case MAT_MATTE:
				Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &pdf, &f, &shadeN, u0, u1
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
                               , &path->specularBounce
#endif
                               );
				break;
#endif
            case MAT_BLINNPHONG:
                BlinnPhong_Sample_f(&hitPointMat->param.blinnPhong, &wo, &wi, &pdf, &f, &shadeN, u0, u1);
                break;
                
#if defined(PARAM_ENABLE_MAT_AREALIGHT)
			case MAT_AREALIGHT: {
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
				if (path->specularBounce) {
#endif
					Spectrum radiance;
					AreaLight_Le(&hitPointMat->param.areaLight, &wo, &N, &radiance);
                    radiance.r *= throughput.x;
                    radiance.g *= throughput.y;
                    radiance.b *= throughput.z;
                    
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
					radiance.r += path->accumRadiance.x;
					radiance.g += path->accumRadiance.y;
					radiance.b += path->accumRadiance.z;
#endif
                    
					TerminatePath(path, ray, frameBuffer, &seed, &radiance);
                    
					// Save the seed
					path->seed.s1 = seed.s1;
					path->seed.s2 = seed.s2;
					path->seed.s3 = seed.s3;
                    
					return;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
				}
				break;
#endif
			}
#endif
			case MAT_NULL:
				wi = rayDir;
				pdf = 1.f;
				f.r = 1.f;
				f.g = 1.f;
				f.b = 1.f;
                
				// I have also to restore the original throughput
				throughput = path->throughput;
				break;
                
			default:
				// Houston, we have a problem...
				pdf = 0.f;
				break;
		}
        
		const uint pathDepth = path->depth + 1;
        const float invPdf = ((pdf <= 0.f) || (pathDepth >= PARAM_MAX_PATH_DEPTH)) ? 0.f : (1.f / pdf);
        throughput.x *= f.r * invPdf;
		throughput.y *= f.g * invPdf;
		throughput.z *= f.b * invPdf;
        
        // Russian roulette
        const float rrProb = max(max(throughput.x, max(throughput.y, throughput.z)), (float)PARAM_RR_CAP);
        const float rrSample = RndFloatValue(&seed);
        const float invRRProb = (pathDepth > PARAM_RR_DEPTH) ? ((rrProb >= rrSample) ? 0.f : (1.f / rrProb)) : 1.f;
        throughput.x *= invRRProb;
        throughput.y *= invRRProb;
        throughput.z *= invRRProb;
        
		if ((throughput.x <= 0.f) && (throughput.y <= 0.f) && (throughput.z <= 0.f)) {
			Spectrum radiance;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
			radiance.r = path->accumRadiance.x;
			radiance.g = path->accumRadiance.y;
			radiance.b = path->accumRadiance.z;
#else
			radiance.r = 0.f;
			radiance.g = 0.f;
			radiance.b = 0.f;
#endif
            
			TerminatePath(path, ray, frameBuffer, &seed, &radiance);
		} else {
			path->throughput = throughput;
            
			// Setup next ray
			ray->o = hitPoint;
			ray->d = wi;
			ray->mint = PARAM_RAY_EPSILON;
			ray->maxt = FLT_MAX;
            
			path->depth = pathDepth;
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
			path->state = PATH_STATE_NEXT_VERTEX;
#endif
		}
	}
    else {
		Spectrum radiance;
        
		radiance.r = 0.f;
		radiance.g = 0.f;
		radiance.b = 0.f;
        
#if defined(PARAM_DIRECT_LIGHT_SAMPLING)
		radiance.r += path->accumRadiance.x;
		radiance.g += path->accumRadiance.y;
		radiance.b += path->accumRadiance.z;
#endif
        
		TerminatePath(path, ray, frameBuffer, &seed, &radiance);
	}
    
	// Save the seed
	path->seed.s1 = seed.s1;
	path->seed.s2 = seed.s2;
	path->seed.s3 = seed.s3;
}
