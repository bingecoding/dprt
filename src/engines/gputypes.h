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


#ifndef GPUTYPES_H
#define GPUTYPES_H

#include "raytracer.h"
#include "spectrum.h"
#include "ray.h"
#include "normal.h"

#define BLOCKWH 4

typedef struct {
    cl_float3 estimatedRadiance;
    cl_float3 boundMaterialTerm;
    cl_float errorBound;
    cl_int id;
} HeapNode;

typedef struct {
    cl_float3 intensity;
} SampleGPU;

typedef struct {
    unsigned int s1, s2, s3;
} Seed;

typedef struct {
    Spectrum throughput;
    unsigned int depth, pixelIndex, subpixelIndex;
    Seed seed;
} Path;

typedef struct {
    
    Ray pathRay;
    RayHit pathHit;
    
    cl_float3 lightRadiance;
    cl_float3 accumRadiance;
    cl_float3 throughput;
    Seed seed;
    
    unsigned int depth, pixelIndex, subpixelIndex;
    uint lightCount;
    int specularBounce;
    int state;
    int clusterIndex;
    
} PathDL;

typedef struct {
    
    unsigned int depth;
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
    Normal n;
} VPL;

typedef struct {
    Spectrum c;
    unsigned int count;
} PixelGPU;


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
} MaterialGPU;

typedef struct {
    Point v0, v1, v2;
    Normal normal;
    float area;
    float gain_r, gain_g, gain_b;
} TriangleLightGPU;

typedef struct {
    unsigned int rgbOffset, alphaOffset;
    unsigned int width, height;
} TexMap;

#endif
