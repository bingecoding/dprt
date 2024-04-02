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

#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <algorithm>

inline void ConcentricSampleDisk(const float u1, const float u2,
                                 float *dx, float *dy)
{
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
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}

inline Vector CosineSampleHemisphere(const float u1, const float u2) {
    Vector ret;
    ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
    ret.z = sqrtf(Max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y));
    
    return ret;
}

inline Vector UniformSampleSphere(const float u1, const float u2) {
    float z = 1.f - 2.f * u1;
    float r = sqrtf(Max(0.f, 1.f - z * z));
    float phi = 2.f * M_PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    
    return Vector(x, y, z);
}

struct Distribution1D {
    // Distribution1D Public Methods
    Distribution1D(const float *f, int n) {
        count = n;
        func = new float[n];
        memcpy(func, f, n*sizeof(float));
        cdf = new float[n+1];
        // Compute integral of step function at $x_i$
        cdf[0] = 0.;
        for (int i = 1; i < count+1; ++i)
            cdf[i] = cdf[i-1] + func[i-1] / n;
        
        // Transform step function integral into CDF
        funcInt = cdf[count];
        if (funcInt == 0.f) {
            for (int i = 1; i < n+1; ++i)
                cdf[i] = float(i) / float(n);
        }
        else {
            for (int i = 1; i < n+1; ++i)
                cdf[i] /= funcInt;
        }
    }
    ~Distribution1D() {
        delete[] func;
        delete[] cdf;
    }
    float SampleContinuous(float u, float *pdf, int *off = NULL) const {
        // Find surrounding CDF segments and _offset_
        float *ptr = std::upper_bound(cdf, cdf+count+1, u);
        int offset = std::max(0, int(ptr-cdf-1));
        if (off) *off = offset;
        assert(offset < count);
        assert(u >= cdf[offset] && u < cdf[offset+1]);
        
        // Compute offset along CDF segment
        float du = (u - cdf[offset]) / (cdf[offset+1] - cdf[offset]);
        assert(!isnan(du));
        
        // Compute PDF for sampled offset
        if (pdf) *pdf = func[offset] / funcInt;
        
        // Return $x\in{}[0,1)$ corresponding to sample
        return (offset + du) / count;
    }
    int SampleDiscrete(float u, float *pdf) const {
        // Find surrounding CDF segments and _offset_
        float *ptr = std::upper_bound(cdf, cdf+count+1, u);
        int offset = std::max(0, int(ptr-cdf-1));
        assert(offset < count);
        assert(u >= cdf[offset] && u < cdf[offset+1]);
        if (pdf) *pdf = func[offset] / (funcInt * count);
        return offset;
    }
private:
    // Distribution1D Private Data
    float *func, *cdf;
    float funcInt;
    int count;
};


#endif
