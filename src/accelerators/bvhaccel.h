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

#ifndef BVHACCEL_H
#define BVHACCEL_H

#include <vector>

#include "accelerators/accelerator.h"

struct BVHAccelTreeNode {
    BBox bbox;
    unsigned int primitive;
    BVHAccelTreeNode *leftChild;
    BVHAccelTreeNode *rightSibling;
};

struct BVHAccelArrayNode {
    BBox bbox;
    unsigned int primitive;
    unsigned int skipIndex;
};

class BVHAccel : public Accelerator {
public:
    
    BVHAccel(const unsigned int treetype, const int csamples, const int icost,
             const int tcost, const float ebonus);
    ~BVHAccel();
    
    AcceleratorType GetType() const { return ACCEL_BVH; }
    
    void Init(const std::deque<TriangleMesh *> &meshes,
              const unsigned int totalVertexCount,
              const unsigned int totalTriangleCount);
    
    const TriangleMeshID GetMeshID(const unsigned int index) const {
        return m_preprocessedMeshIDs[index]; }
    const TriangleMeshID *GetMeshIDTable() const {
        return m_preprocessedMeshIDs;
    }
    
    const TriangleID GetMeshTriangleID(const unsigned int index) const {
        return m_preprocessedMeshTriangleIDs[index];
    }
    const TriangleID *GetMeshTriangleIDTable() const {
        return m_preprocessedMeshTriangleIDs;
    }
    
    bool Intersect(const Ray *ray, RayHit *hit) const;
    bool IntersectP(const Ray *ray) const;
    
    const TriangleMesh *GetPreprocessedMesh() const {
        return m_preprocessedMesh;
    }
    
    friend class OpenCLIntersectionDevice;
    
private:
    
    BVHAccelTreeNode *BuildHierarchy(std::vector<BVHAccelTreeNode *> &list,
                                     unsigned int begin, unsigned int end,
                                     unsigned int axis);
    void FindBestSplit(std::vector<BVHAccelTreeNode *> &list, unsigned int begin,
                       unsigned int end, float *splitValue, unsigned int *bestAxis);
    
    unsigned int BuildArray(BVHAccelTreeNode *node, unsigned int offset);
    void FreeHierarchy(BVHAccelTreeNode *node);
    
    
    unsigned int m_treeType;
    int m_costSamples, m_isectCost, m_traversalCost;
    float m_emptyBonus;
    unsigned int m_nNodes;
    BVHAccelArrayNode *m_bvhTree;
    
    TriangleMesh *m_preprocessedMesh;
    TriangleMeshID *m_preprocessedMeshIDs;
    TriangleID *m_preprocessedMeshTriangleIDs;
    
    bool m_initialized;
    
};


#endif
