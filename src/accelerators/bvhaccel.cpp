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

#include "raytracer.h"

#include "accelerators/bvhaccel.h"

BVHAccel::BVHAccel(const unsigned int treetype,
                   const int csamples, const int icost,
                   const int tcost, const float ebonus) :

m_costSamples(csamples), m_isectCost(icost),
m_traversalCost(tcost), m_emptyBonus(ebonus)
{
    
    
    if(treetype <= 2) m_treeType = 2;
    else if (treetype <=4) m_treeType = 4;
    else m_treeType = 8;
    
    m_initialized = false;
    
}

void BVHAccel::Init(const std::deque<TriangleMesh *> &meshes,
                    const unsigned int totalVertexCount,
                    const unsigned int totalTriangleCount) {

    
    assert(!m_initialized);
    
    m_preprocessedMesh = TriangleMesh::Merge(totalVertexCount,
                                             totalTriangleCount,
                                             meshes, &m_preprocessedMeshIDs,
                                             &m_preprocessedMeshTriangleIDs);
    assert(m_preprocessedMesh->GetTotalVertexCount() == totalVertexCount);
    assert(m_preprocessedMesh->GetTotalTriangleCount() == totalTriangleCount);
    
    RT_LOG("Total vertices memory usage: " <<
           totalVertexCount * sizeof(Point) /1024 << "Kbytes");
	RT_LOG("Total triangles memory usage: " <<
           totalTriangleCount * sizeof(Triangle) / 1024 << "Kbytes");
    
    
    const Point *v = m_preprocessedMesh->GetVertices();
    const Triangle *p = m_preprocessedMesh->GetTriangles();
    
    std::vector<BVHAccelTreeNode *> bvList;
    for(unsigned int i=0; i < totalTriangleCount; ++i) {
        BVHAccelTreeNode *ptr = new BVHAccelTreeNode();
        ptr->bbox = p[i].WorldBound(v);
        // Expand bbox minutely to make sure rays collide
        ptr->bbox.Expand(RAY_EPSILON);
        ptr->primitive = i;
        ptr->leftChild = NULL;
        ptr->rightSibling = NULL;
        bvList.push_back(ptr);
    }
    
    RT_LOG("Building Bounding Volume Hierarchy, primitives:" <<
           totalTriangleCount);
    
    m_nNodes = 0;
    BVHAccelTreeNode *rootNode = BuildHierarchy(bvList, 0, bvList.size(), 2);
    
    RT_LOG("Preprocessing Bounding Volume Hierarchy, total nodes: " << m_nNodes);
    
    m_bvhTree = new BVHAccelArrayNode[m_nNodes];
    BuildArray(rootNode, 0);
    FreeHierarchy(rootNode);
    
    RT_LOG("Total BVH memory usage: " <<
           m_nNodes * sizeof(BVHAccelArrayNode) / 1024 << "Kbytes");
	RT_LOG("Finished building Bounding Volume Hierarchy array");
    
	m_initialized = true;
    
}

BVHAccel::~BVHAccel() {
    if(m_initialized) {
        m_preprocessedMesh->Delete();
        delete m_preprocessedMesh;
        delete [] m_preprocessedMeshIDs;
        delete [] m_preprocessedMeshTriangleIDs;
        delete m_bvhTree;
    }
}

void BVHAccel::FreeHierarchy(BVHAccelTreeNode *node) {
    if(node) {
        FreeHierarchy(node->leftChild);
        FreeHierarchy(node->rightSibling);
        
        delete node;
    }
}

// Comparator for each axis

bool bvh_ltf_x(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.x + n->bbox.pMin.x < v;
}

bool bvh_ltf_y(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.y + n->bbox.pMin.y < v;
}

bool bvh_ltf_z(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.z + n->bbox.pMin.z < v;
}

bool (* const bvh_ltf[3])(BVHAccelTreeNode *n, float v) = {bvh_ltf_x, bvh_ltf_y, bvh_ltf_z};

BVHAccelTreeNode *BVHAccel::BuildHierarchy(std::vector<BVHAccelTreeNode *> &list,
                                           unsigned int begin, unsigned int end,
                                           unsigned int axis) {
    
    unsigned int splitAxis = axis;
    float splitValue;
    
    m_nNodes += 1;
    if(end - begin == 1) {
        return list[begin];
    }
    
    BVHAccelTreeNode *parent = new BVHAccelTreeNode();
    parent->primitive = 0xffffffffu;
    parent->leftChild = NULL;
    parent->rightSibling = NULL;
    
    std::vector<unsigned int> splits;
    splits.reserve(m_treeType + 1);
    splits.push_back(begin);
    splits.push_back(end);
    // Calculate splits, according to tree size
    for(unsigned int i = 2; i <= m_treeType; i*=2) {
        for(unsigned int j=0, offset=0; (j+offset < i) && (splits.size() > j+1); j+=2) {
            if (splits[j+1] - splits[j] < 2) {
				j--;
				offset++;
				continue; // Less than 2 elements, no need to split
			}
            
            FindBestSplit(list, splits[j], splits[j + 1], &splitValue, &splitAxis);
            
			std::vector<BVHAccelTreeNode *>::iterator it =
            partition(list.begin() + splits[j], list.begin() + splits[j + 1], std::bind2nd(std::ptr_fun(bvh_ltf[splitAxis]), splitValue));
			unsigned int middle = distance(list.begin(), it);
			middle = Max(splits[j] + 1, Min(splits[j + 1] - 1, middle)); // Make sure coincidental BBs are still split
			splits.insert(splits.begin() + j + 1, middle);
		}
	}
    
	BVHAccelTreeNode *child, *lastChild;
	// Left Child
	child = BuildHierarchy(list, splits[0], splits[1], splitAxis);
	parent->leftChild = child;
	parent->bbox = child->bbox;
	lastChild = child;
    
	// Add remaining children
	for (unsigned int i = 1; i < splits.size() - 1; i++) {
		child = BuildHierarchy(list, splits[i], splits[i + 1], splitAxis);
		lastChild->rightSibling = child;
		parent->bbox = Union(parent->bbox, child->bbox);
		lastChild = child;
	}
    
	return parent;
}

void BVHAccel::FindBestSplit(std::vector<BVHAccelTreeNode *> &list, unsigned int begin, unsigned int end, float *splitValue, unsigned int *bestAxis) {
	if (end - begin == 2) {
		// Trivial case with two elements
		*splitValue = (list[begin]->bbox.pMax[0] + list[begin]->bbox.pMin[0] +
                       list[end - 1]->bbox.pMax[0] + list[end - 1]->bbox.pMin[0]) / 2;
		*bestAxis = 0;
	} else {
		// Calculate BBs mean center (times 2)
		Point mean2(0, 0, 0), var(0, 0, 0);
		for (unsigned int i = begin; i < end; i++)
			mean2 += list[i]->bbox.pMax + list[i]->bbox.pMin;
		mean2 /= end - begin;
        
		// Calculate variance
		for (unsigned int i = begin; i < end; i++) {
			Vector v = list[i]->bbox.pMax + list[i]->bbox.pMin - mean2;
			v.x *= v.x;
			v.y *= v.y;
			v.z *= v.z;
			var += v;
		}
		// Select axis with more variance
		if (var.x > var.y && var.x > var.z)
			*bestAxis = 0;
		else if (var.y > var.z)
			*bestAxis = 1;
		else
			*bestAxis = 2;
        
		if (m_costSamples > 1) {
			BBox nodeBounds;
			for (unsigned int i = begin; i < end; i++)
				nodeBounds = Union(nodeBounds, list[i]->bbox);
            
			Vector d = nodeBounds.pMax - nodeBounds.pMin;
			const float invTotalSA = 1.f / nodeBounds.SurfaceArea();
            
			// Sample cost for split at some points
			float increment = 2 * d[*bestAxis] / (m_costSamples + 1);
			float bestCost = INFINITY;
			for (float splitVal = 2 * nodeBounds.pMin[*bestAxis] + increment; splitVal < 2 * nodeBounds.pMax[*bestAxis]; splitVal += increment) {
				int nBelow = 0, nAbove = 0;
				BBox bbBelow, bbAbove;
				for (unsigned int j = begin; j < end; j++) {
					if ((list[j]->bbox.pMax[*bestAxis] + list[j]->bbox.pMin[*bestAxis]) < splitVal) {
						nBelow++;
						bbBelow = Union(bbBelow, list[j]->bbox);
					} else {
						nAbove++;
						bbAbove = Union(bbAbove, list[j]->bbox);
					}
				}
				const float pBelow = bbBelow.SurfaceArea() * invTotalSA;
				const float pAbove = bbAbove.SurfaceArea() * invTotalSA;
				float eb = (nAbove == 0 || nBelow == 0) ? m_emptyBonus : 0.f;
				float cost = m_traversalCost + m_isectCost * (1.f - eb) * (pBelow * nBelow + pAbove * nAbove);
				// Update best split if this is lowest cost so far
				if (cost < bestCost) {
					bestCost = cost;
					*splitValue = splitVal;
				}
			}
		} else {
			// Split in half around the mean center
			*splitValue = mean2[*bestAxis];
		}
	}
}

unsigned int BVHAccel::BuildArray(BVHAccelTreeNode *node, unsigned int offset) {
	// Build array by recursively traversing the tree depth-first
	while (node) {
		BVHAccelArrayNode *p = &m_bvhTree[offset];
        
		p->bbox = node->bbox;
		p->primitive = node->primitive;
		offset = BuildArray(node->leftChild, offset + 1);
		p->skipIndex = offset;
        
		node = node->rightSibling;
	}
    
	return offset;
}

bool BVHAccel::Intersect(const Ray *ray, RayHit *rayHit) const {
	assert (m_initialized);
    
	unsigned int currentNode = 0; // Root Node
	unsigned int stopNode = m_bvhTree[0].skipIndex; // Non-existent
	bool hit = false;
	rayHit->t = std::numeric_limits<float>::infinity();
	rayHit->index = 0xffffffffu;
	RayHit triangleHit;
    
	const Point *vertices = m_preprocessedMesh->GetVertices();
	const Triangle *triangles = m_preprocessedMesh->GetTriangles();
    
	while (currentNode < stopNode) {
		if (m_bvhTree[currentNode].bbox.IntersectP(*ray)) {
            
			if (m_bvhTree[currentNode].primitive != 0xffffffffu) {
				if (triangles[m_bvhTree[currentNode].primitive].Intersect(*ray, vertices, &triangleHit)) {
					hit = true; // Continue testing for closer intersections
					if (triangleHit.t < rayHit->t) {
						rayHit->t = triangleHit.t;
						rayHit->b1 = triangleHit.b1;
						rayHit->b2 = triangleHit.b2;
						rayHit->index = m_bvhTree[currentNode].primitive;
					}
				}
			}
            
			currentNode++;
		} else
			currentNode = m_bvhTree[currentNode].skipIndex;
	}
    
	return hit;
}

bool BVHAccel::IntersectP(const Ray *ray) const {
    assert (m_initialized);
    
    unsigned int currentNode = 0; // Root Node
    unsigned int stopNode = m_bvhTree[0].skipIndex; // Non-existent
    bool hit = false;
    RayHit triangleHit;
    
    const Point *vertices = m_preprocessedMesh->GetVertices();
    const Triangle *triangles = m_preprocessedMesh->GetTriangles();
    
    while (currentNode < stopNode) {
        if (m_bvhTree[currentNode].bbox.IntersectP(*ray)) {
            if (m_bvhTree[currentNode].primitive != 0xffffffffu) {
                if (triangles[m_bvhTree[currentNode].primitive].Intersect(*ray, vertices, &triangleHit)) {
                    hit = true; // Continue testing for closer intersections
                }
            }
            currentNode++;
        } else
        currentNode = m_bvhTree[currentNode].skipIndex;
    }
    
    return hit;
}
