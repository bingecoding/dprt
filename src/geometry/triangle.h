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

#ifndef TRIANGLE_H_
#define TRIANGLE_H_

inline void UniformSamlpleTriangle(const float u0, const float u1, float *u, float *v ) {
    float su1 = sqrtf(u0);
    *u = 1.f - su1;
    *v = u1 * su1;
}

class Triangle {
    
public:
    
    Triangle() {}
    
    Triangle(const unsigned int v0, const unsigned int v1, const unsigned int v2) {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
    }
    
    BBox WorldBound(const Point *verts) const {
        const Point &p0 = verts[v[0]];
        const Point &p1 = verts[v[1]];
        const Point &p2 = verts[v[2]];
        
        return Union(BBox(p0, p1), p2);
    }
    
    bool Intersect(const Ray &ray, const Point *verts, RayHit *triangleHit) const {
        
        const Point &p0 = verts[v[0]];
        const Point &p1 = verts[v[1]];
        const Point &p2 = verts[v[2]];
        
        const Vector e1 = p1 - p0;
        const Vector e2 = p2 - p0;
        const Vector s1 = Cross(ray.d, e2);
        
        const float divisor = Dot(s1, e1);
        if(divisor == 0.f) {
            return false;
        }
        
        const float invDivisor = 1.f / divisor;
        
        // Compute 1st barycentric coordinate
        const Vector d = ray.o - p0;
        const float b1 = Dot(d, s1) * invDivisor;
        if(b1 < 0.f) {
            return false;
        }
        
        // Compute 2nd barycentric coordinate
        const Vector s2 = Cross(d, e1);
        const float b2 = Dot(ray.d, s2) * invDivisor;
        if(b2 < 0.f) {
            return false;
        }
        
        const float b0 = 1.f - b1 - b2;
        if(b0 < 0.f) {
            return false;
        }
        
        // Compute t to intersection point
        const float t = Dot(e2, s2) * invDivisor;
        if( t < ray.mint || t > ray.maxt) {
            return false;
        }
        
        triangleHit->t = t;
        triangleHit->b1 = b1;
        triangleHit->b2 = b2;
        
        return true;
    }
    
    float Area(const Point *verts) const {
        const Point &p0 = verts[v[0]];
        const Point &p1 = verts[v[1]];
        const Point &p2 = verts[v[2]];
        
        return 0.5f * Cross(p1 - p0, p2 - p0).Length();
    }
    
    void Sample(const Point *verts, const float u0,
                const float u1, Point *p,
                float *b0, float *b1, float *b2) const {
        
        UniformSamlpleTriangle(u0, u1, b0, b1);
        
        const Point &p0 = verts[v[0]];
        const Point &p1 = verts[v[1]];
        const Point &p2 = verts[v[2]];
        
        *b2 = 1.f - (*b0) - (*b1);
        *p = (*b0) * p0 + (*b1) * p1 + (*b2) * p2;
        
    }
                
    static float Area(const Point &p0, const Point &p1, const Point &p2) {
		return 0.5f * Cross(p1 - p0, p2 - p0).Length();
	}
    
    unsigned int v[3];
    
};

inline std::ostream & operator<<(std::ostream &os, const Triangle &tri) {
	os << "Triangle[" << tri.v[0] << ", " << tri.v[1] << ", " << tri.v[2] << "]";
	return os;
}

#endif
