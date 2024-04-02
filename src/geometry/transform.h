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

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "geometry/normal.h"
#include "geometry/bbox.h"
#include "geometry/matrix4x4.h"

// Transform Declarations
class Transform {
public:
	// Transform Public Methods
	Transform() {
		// use the preallocated identity matrix because it will never be changed
		m = mInv = MAT_IDENTITY;
	}
    
	Transform(float mat[4][4]) {
		Matrix4x4 o(mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                    mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                    mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                    mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
		m = o;
		mInv = m.Inverse();
	}
    
	Transform(const Matrix4x4 &mat) {
		m = mat;
		mInv = m.Inverse();
	}
    
	Transform(const Matrix4x4 &mat,
              const Matrix4x4 &minv) {
		m = mat;
		mInv = minv;
	}
    
	friend std::ostream &operator<<(std::ostream &, const Transform &);
    
	Transform GetInverse() const {
		return Transform(mInv, m);
	}
    
	Matrix4x4 GetMatrix() const {
		return m;
	}
	bool HasScale() const;
	inline Point operator()(const Point &pt) const;
	inline void operator()(const Point &pt, Point *ptrans) const;
	inline Vector operator()(const Vector &v) const;
	inline void operator()(const Vector &v, Vector *vt) const;
	inline Normal operator()(const Normal &) const;
	inline void operator()(const Normal &, Normal *nt) const;
	inline Ray operator()(const Ray &r) const;
	inline void operator()(const Ray &r, Ray *rt) const;
	BBox operator()(const BBox &b) const;
	Transform operator*(const Transform &t2) const;
	bool SwapsHandedness() const;
    
private:
	// Transform Private Data
	Matrix4x4 m, mInv;
    
	static const Matrix4x4 MAT_IDENTITY;
};

inline Point Transform::operator()(const Point &pt) const {
	const float x = pt.x, y = pt.y, z = pt.z;
	const float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
	const float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
	const float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
	const float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    
	//	BOOST_ASSERT(wp != 0);
	if (wp == 1.)
        return Point(xp, yp, zp);
	else
        return Point(xp, yp, zp) / wp;
}

inline void Transform::operator()(const Point &pt, Point *ptrans) const {
	const float x = pt.x, y = pt.y, z = pt.z;
	ptrans->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
	ptrans->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
	ptrans->z = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
	const float w = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
	if (w != 1.) *ptrans /= w;
}

inline Vector Transform::operator()(const Vector &v) const {
	const float x = v.x, y = v.y, z = v.z;
	return Vector(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                  m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                  m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

inline void Transform::operator()(const Vector &v,
                                  Vector *vt) const {
	const float x = v.x, y = v.y, z = v.z;
	vt->x = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z;
	vt->y = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z;
	vt->z = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z;
}

inline Normal Transform::operator()(const Normal &n) const {
	const float x = n.x, y = n.y, z = n.z;
	return Normal(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
                  mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
                  mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
}

inline void Transform::operator()(const Normal &n,
                                  Normal *nt) const {
	const float x = n.x, y = n.y, z = n.z;
	nt->x = mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z;
	nt->y = mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z;
	nt->z = mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z;
}

inline bool Transform::SwapsHandedness() const {
	const float det = ((m.m[0][0] *
                        (m.m[1][1] * m.m[2][2] -
                         m.m[1][2] * m.m[2][1])) -
                       (m.m[0][1] *
                        (m.m[1][0] * m.m[2][2] -
                         m.m[1][2] * m.m[2][0])) +
                       (m.m[0][2] *
                        (m.m[1][0] * m.m[2][1] -
                         m.m[1][1] * m.m[2][0])));
	return det < 0.f;
}

inline Ray Transform::operator()(const Ray &r) const {
	Ray ret((*this)(r.o), (*this)(r.d), r.mint, r.maxt);
    
	return ret;
}

inline void Transform::operator()(const Ray &r,
                                  Ray *rt) const {
	(*this)(r.o, &rt->o);
	(*this)(r.d, &rt->d);
	rt->mint = r.mint;
	rt->maxt = r.maxt;
}

Transform Translate(const Vector &delta);
Transform Scale(float x, float y, float z);
Transform RotateX(float angle);
Transform RotateY(float angle);
Transform RotateZ(float angle);
Transform Rotate(float angle, const Vector &axis);
Transform LookAt(const Point &pos, const Point &look, const Vector &up);
Transform Orthographic(float znear, float zfar);
Transform Perspective(float fov, float znear, float zfar);
void TransformAccordingNormal(const Normal &nn, const Vector &woL, Vector *woW);

#endif
