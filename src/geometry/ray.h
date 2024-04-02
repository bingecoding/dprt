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

#ifndef RAY_H
#define RAY_H

#include "geometry/point.h"


extern float RAY_EPSILON;

class Ray {
    
public:
    
    Ray() : maxt(std::numeric_limits<float>::infinity()) {
        mint = 1e-4f;
    }
    
    Ray(const Point &origin, const Vector &direction) :
    o(origin), d(direction), maxt(std::numeric_limits<float>::infinity()) {
        mint = RAY_EPSILON;
    }
    
    Ray(const Point &origin, const Vector &direction,
        float start, float end = std::numeric_limits<float>::infinity())
    : o(origin), d(direction), mint(start), maxt(end) {}
    
    Point operator()(float t) const { return o + d * t; }
    void GetDirectionSigns(int signs[3]) const {
        signs[0] = d.x < 0.f;
        signs[1] = d.y < 0.f;
        signs[2] = d.z < 0.f;
    }
    
    Point o;
    Vector d;
    mutable float maxt, mint;
  
};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "Ray[" << r.o << ", " << r.d << ", " << r.mint << "," << r.maxt << "]";
    return os;
}

class RayHit {
public:
    float t;
    float b1, b2; // Barycentric coordinates of the hit point
    unsigned int index;
    
    void SetMiss() { index = 0xffffffffu; };
    bool Miss() const { return (index == 0xffffffffu); };
};


#endif
