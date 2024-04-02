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

#include "utils/trianglemesh.h"
#include "raytracer.h"

BBox TriangleMesh::GetBBox() const {
	BBox bbox;
	for (unsigned int i = 0; i < m_vertCount; ++i)
		bbox = Union(bbox, m_vertices[i]);
    
	return bbox;
}

TriangleMesh *TriangleMesh::Merge(const unsigned int totalVertexCount,
                                  const unsigned int totalTriangleCount,
                                  const std::deque<TriangleMesh *> &meshes,
                                  TriangleMeshID **preprocessMeshIDs,
                                  TriangleID **preprocessMeshTriangleIDs)
{
    assert (totalVertexCount > 0);
	assert (totalTriangleCount > 0);
	assert (meshes.size() > 0);
    
    Point *v = new Point[totalVertexCount];
    Triangle *i = new Triangle[totalTriangleCount];
    
    if(preprocessMeshIDs) {
        *preprocessMeshIDs = new TriangleMeshID[totalTriangleCount];
    }
    if (preprocessMeshTriangleIDs) {
        *preprocessMeshTriangleIDs = new TriangleID[totalTriangleCount];
    }
    
    unsigned int vIndex = 0;
    unsigned int iIndex = 0;
    TriangleMeshID currentID = 0;
    for(std::deque<TriangleMesh *>::const_iterator m = meshes.begin(); m < meshes.end(); m++) {
        const Triangle *tris;
       
        const TriangleMesh *mesh = *m;
        
        // Copy mesh vertices
        for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); j++)
            v[vIndex + j] = mesh->GetVertex(j);
        
        tris = mesh->GetTriangles();
        
        // Translate mesh indices
        for(unsigned int j=0; j < mesh->GetTotalTriangleCount(); j++) {
            i[iIndex].v[0] = tris[j].v[0] + vIndex;
            i[iIndex].v[1] = tris[j].v[1] + vIndex;
            i[iIndex].v[2] = tris[j].v[2] + vIndex;
            
            if(preprocessMeshIDs) {
                (*preprocessMeshIDs)[iIndex] = currentID;
            }
            if(preprocessMeshTriangleIDs) {
                (*preprocessMeshTriangleIDs)[iIndex] = j;
            }
            
            ++iIndex;
        }
        
        vIndex += mesh->GetTotalVertexCount();
        if(preprocessMeshIDs) {
            // this is to avoid compiler warning
            currentID = currentID + 1;
        }
        
    }
    
    return new TriangleMesh(totalVertexCount, totalTriangleCount, v, i);
}

TriangleMesh *TriangleMesh::LoadTriangleMesh(const tinyobj::shape_t &shape)
{
    const unsigned int vertexCount = shape.mesh.positions.size() / 3;
	const unsigned int triangleCount = shape.mesh.indices.size() / 3;
    
    Point *vertices = new Point[vertexCount];
	Triangle *triangles = new Triangle[triangleCount];
    
    for(unsigned int i=0; i < vertexCount; i++){
        float x = shape.mesh.positions[3*i+0];
        float y = shape.mesh.positions[3*i+1];
        float z = shape.mesh.positions[3*i+2];
        vertices[i] = Point(x, y ,z);
    }
    
    for(unsigned int i=0; i < triangleCount; i++){
        unsigned int v0 = shape.mesh.indices[3*i+0];
        unsigned int v1 = shape.mesh.indices[3*i+1];
        unsigned int v2 = shape.mesh.indices[3*i+2];
        triangles[i] = Triangle(v0, v1, v2);
    }
    
    // Texture coordinates
    UV *vertUV = NULL;
    if (shape.mesh.texcoords.size() > 0) {
        vertUV = new UV[vertexCount];
        for (size_t k = 0; k < vertexCount; k++) {
            float u = shape.mesh.texcoords[2*k+0];
            float v = shape.mesh.texcoords[2*k+1];
            vertUV[k] = UV(u,v);
        }
    }
    
	Normal *vertNormals = new Normal[vertexCount];;
	Spectrum *vertColors = NULL;
    for (unsigned int i = 0; i < vertexCount; ++i)
        vertNormals[i] = Normal(0.f, 0.f, 0.f);
    for (unsigned int i = 0; i < triangleCount; ++i) {
        const Vector e1 = vertices[triangles[i].v[1]] - vertices[triangles[i].v[0]];
        const Vector e2 = vertices[triangles[i].v[2]] - vertices[triangles[i].v[0]];
        const Normal N = Normal(Normalize(Cross(e1, e2)));
        vertNormals[triangles[i].v[0]] += N;
        vertNormals[triangles[i].v[1]] += N;
        vertNormals[triangles[i].v[2]] += N;
    }
    int printedWarning = 0;
    for (unsigned int i = 0; i < vertexCount; ++i) {
        vertNormals[i] = Normalize(vertNormals[i]);
        // Check for degenerate triangles/normals, they can freeze the GPU
        if (isnan(vertNormals[i].x) || isnan(vertNormals[i].y) || isnan(vertNormals[i].z)) {
            if (printedWarning < 15) {
                RT_LOG("The model contains a degenerate normal (index " << i << ")");
                ++printedWarning;
            } else if (printedWarning == 15) {
                RT_LOG("The model contains more degenerate normals");
                ++printedWarning;
            }
            vertNormals[i] = Normal(0.f, 0.f, 1.f);
        }
    }
    
    return new TriangleMesh(vertexCount, triangleCount, vertices, triangles,
                            vertNormals, vertUV);
    
}
