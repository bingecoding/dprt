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

#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "utils.h"

static const int rgbChannels = 3;

class Spectrum {
public:
    
    Spectrum(const float _r = 0.f, const float _g = 0.f, const float _b = 0.f)
    : r(_r), g(_g), b(_b) {
    }
    
    Spectrum operator+(const Spectrum &v) const {
        return Spectrum(r + v.r, g + v.g, b + v.b);
    }

    Spectrum operator-(const Spectrum &v) const {
        return Spectrum(r - v.r, g - v.g, b - v.b);
    }
    
    Spectrum operator*(const Spectrum &v) const {
        return Spectrum(r * v.r, g * v.g, b * v.b);
    }
    
    Spectrum operator*(float f) const
    {
        return Spectrum(f*r, f*g, f * b);
    }
    
    Spectrum & operator*=(const Spectrum &v) {
        r *= v.r;
        g *= v.g;
        b *= v.b;
        return *this;
    }
    
    Spectrum & operator*=(float f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }
    
    Spectrum & operator+=(const Spectrum &v) {
        r += v.r;
        g += v.g;
        b += v.b;
        return *this;
    }
    
    Spectrum & operator-=(const Spectrum &v) {
        r -= v.r;
        g -= v.g;
        b -= v.b;
        return *this;
    }
    
    Spectrum operator/(const Spectrum &s) const {
        return Spectrum(r / s.r, g / s.g, b / s.b);
    }
    
    Spectrum & operator/=(const Spectrum &s) {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }
    
    Spectrum operator/(float f) const {
        float inv = 1.f / f;
        return Spectrum(r * inv, g * inv, b * inv);
    }
    
    Spectrum & operator/=(float f) {
        float inv = 1.f / f;
        r *= inv;
        g *= inv;
        b *= inv;
        return *this;
    }

    bool Black() const {
        return (r == 0.f) && (g == 0.f) && (b == 0.f);
    }
    
    bool IsNaN() const
    {
        return isnan(r) || isnan(g) || isnan(b);
    }
    
    float Y() const {
        return 0.212671f * r + 0.715160f * g + 0.072169f * b;
    }
    
    float Filter() const {
        return Max<float>(r, Max<float>(g, b));
    }
    
    void Clamp(float low = 0.f) {
        r = r < low ? low : r;
        g = g < low ? low : g;
        b = b < low ? low : b;
       
    }
    
    float r, g, b;
};

inline Spectrum operator*(float f, const Spectrum &v) {
    return v * f;
}

#endif
