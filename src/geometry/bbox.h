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

#ifndef BBOX_H
#define BBOX_H

#include "geometry/ray.h"
#include "geometry/bsphere.h"

class BBox {
public:
	// BBox Public Methods
    
	BBox() {
		pMin = Point(INFINITY, INFINITY, INFINITY);
		pMax = Point(-INFINITY, -INFINITY, -INFINITY);
	}
    
	BBox(const Point &p) : pMin(p), pMax(p) {
	}
    
	BBox(const Point &p1, const Point &p2) {
		pMin = Point(Min(p1.x, p2.x),
                     Min(p1.y, p2.y),
                     Min(p1.z, p2.z));
		pMax = Point(Max(p1.x, p2.x),
                     Max(p1.y, p2.y),
                     Max(p1.z, p2.z));
	}
    
	bool Overlaps(const BBox &b) const {
		bool x = (pMax.x >= b.pMin.x) && (pMin.x <= b.pMax.x);
		bool y = (pMax.y >= b.pMin.y) && (pMin.y <= b.pMax.y);
		bool z = (pMax.z >= b.pMin.z) && (pMin.z <= b.pMax.z);
		return (x && y && z);
	}
    
	bool Inside(const Point &pt) const {
		return (pt.x >= pMin.x && pt.x <= pMax.x &&
				pt.y >= pMin.y && pt.y <= pMax.y &&
				pt.z >= pMin.z && pt.z <= pMax.z);
	}
    
	void Expand(const float delta) {
		pMin -= Vector(delta, delta, delta);
		pMax += Vector(delta, delta, delta);
	}
    
	float Volume() const {
		Vector d = pMax - pMin;
		return d.x * d.y * d.z;
	}
    
	float SurfaceArea() const {
		Vector d = pMax - pMin;
		return 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
	}
    
	int MaximumExtent() const {
		Vector diag = pMax - pMin;
		if (diag.x > diag.y && diag.x > diag.z)
			return 0;
		else if (diag.y > diag.z)
			return 1;
		else
			return 2;
	}
	void BoundingSphere(Point *c, float *rad) const;
	BSphere BoundingSphere() const;
    
	bool IntersectP(const Ray &ray,
                    float *hitt0 = NULL,
                    float *hitt1 = NULL) const;
    
	friend inline std::ostream &operator<<(std::ostream &os, const BBox &b);
	friend BBox Union(const BBox &b, const Point &p);
	friend BBox Union(const BBox &b, const BBox &b2);
    
	// BBox Public Data
	Point pMin, pMax;
};

extern BBox Union(const BBox &b, const Point &p);
extern BBox Union(const BBox &b, const BBox &b2);
extern Point ClosestPointToBBox(Point p, BBox box);

inline std::ostream &operator<<(std::ostream &os, const BBox &b) {
	os << "BBox[" << b.pMin << ", " << b.pMax << "]";
	return os;
}


#endif
