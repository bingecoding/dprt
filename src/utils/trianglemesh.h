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

#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include <deque>

#include <cassert>

#include "transform.h"
#include "triangle.h"
#include "spectrum.h"
#include "uv.h"

#include "utils/tiny_obj_loader.h"

typedef unsigned int TriangleMeshID;
typedef unsigned int TriangleID;

class TriangleMesh {
public:
    
    TriangleMesh(const unsigned int meshVertCount,
                 const unsigned int meshTriCount,
                 Point *meshVertices, Triangle *meshTris,
                 Normal *meshNormals = NULL, UV *meshUV = NULL,
                 Spectrum *meshColors = NULL)
    {
		assert (meshVertCount > 0);
		assert (meshTriCount > 0);
		assert (meshVertices != NULL);
		assert (meshTris != NULL);
        
		m_vertCount = meshVertCount;
		m_triCount = meshTriCount;
		m_vertices = meshVertices;
		m_tris = meshTris;
        
		m_normals = meshNormals;
        m_uvs = meshUV;
        m_colors = meshColors;
    }
    
    virtual ~TriangleMesh() {}
    virtual void Delete() {
        delete[] m_vertices;
        delete[] m_tris;
        delete[] m_normals;
        delete[] m_uvs;
		delete[] m_colors;
    }
    
    unsigned int GetTotalVertexCount() const { return m_vertCount; }
    unsigned int GetTotalTriangleCount() const { return m_triCount; }
    
    bool HasColors() const { return m_colors != NULL; }
    bool HasUVs() const { return m_uvs != NULL; }
    
    BBox GetBBox() const;
    Point GetVertex(const unsigned int vertIndex) const
    {
        return m_vertices[vertIndex];
    }
    float GetTriangleArea(const unsigned int triIndex) const
    {
        return m_tris[triIndex].Area(m_vertices);
    }
    
    Point *GetVertices() const { return m_vertices; }
    Triangle *GetTriangles() const { return m_tris; }
    
    Normal GetNormal(const unsigned int triIndex,
                     const unsigned int vertIndex) const
    {
        return m_normals[m_tris[triIndex].v[vertIndex]];
    }
    
    Normal GetNormal(const unsigned int vertIndex) const
    {
        return m_normals[vertIndex];
    }
    
    UV GetUV(const unsigned int vertIndex) const { return m_uvs[vertIndex]; }
    
    Spectrum GetColor(const unsigned int vertIndex) const {
        return m_colors[vertIndex];
    }
    
    void Sample(const unsigned int index, const float u0, const float u1,
                Point *p, float *b0, float *b1, float *b2) const
    {
		const Triangle &tri = m_tris[index];
		tri.Sample(m_vertices, u0, u1, p, b0, b1, b2);
	}
    
    Normal InterpolateTriNormal(const unsigned int index, const float b1, const float b2) const {
        const Triangle &tri = m_tris[index];
        const float b0 = 1.f - b1 - b2;
        return Normalize(b0 * m_normals[tri.v[0]] + b1 * m_normals[tri.v[1]] + b2 * m_normals[tri.v[2]]);
    }
    
    Spectrum InterpolateTriColor(const unsigned int index, const float b1, const float b2) const {
        const Triangle &tri = m_tris[index];
        const float b0 = 1.f - b1 - b2;
        return b0 * m_colors[tri.v[0]] + b1 * m_colors[tri.v[1]] + b2 * m_colors[tri.v[2]];
    }
    
    static TriangleMesh *Merge(
                               const unsigned int totalVerticesCount,
                               const unsigned int totalIndicesCount,
                               const std::deque<TriangleMesh *> &meshes,
                               TriangleMeshID **preprocessedMeshIDs = NULL,
                               TriangleID **preprocessedMeshTriangleIDs = NULL);
    
    static TriangleMesh *LoadTriangleMesh(const tinyobj::shape_t &shape);
    
protected:
    
    unsigned int m_vertCount;
    unsigned int m_triCount;
    Point *m_vertices;
    Triangle *m_tris;
    
    Normal *m_normals;
    UV *m_uvs;
	Spectrum *m_colors;
    
};


#endif
