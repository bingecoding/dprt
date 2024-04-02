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

#ifndef LIGHTTREE_H
#define LIGHTTREE_H

#define SAMPLESIZE 4
#define ARITY 2

#include "gputypes.h"
#include "sampler.h"
#include "bbox.h"

using namespace std;

class LightTree;
class ClusterPair;
struct KdNode;

struct SampleNode {
    bool sampleSet;
    Vector dir;
    float totalIntensity;
    Spectrum intensity;
};

struct Cluster{
    
    VPL representativeLight;
    // This represents the total light contribution of the cluster
    Spectrum intensity;
    Spectrum geoVisInt;
    Vector vplDir;
    
    Cluster *siblings[ARITY];
    int siblingIDs[ARITY];
    bool isLeaf;
    int ID;
    
    BBox bounds;
    
    Spectrum errorBound;
    Spectrum estimatedRadiance;
    Spectrum repErrorBound;
    Spectrum repEstimatedRadiance;
 
    Spectrum brdf;
    
    SampleNode sampleNode[SAMPLESIZE];
    bool repLight;
    int repId;
    
};

// Compressed version of cluster - prototype for data that will be copied to GPU
struct LightCluster {
        
    LightCluster(const Cluster *const cluster)
    {
        repLightHitPoint = cluster->representativeLight.hitPoint;
        repLightNormal = cluster->representativeLight.n;
        intensity.x = cluster->intensity.r;
        intensity.y = cluster->intensity.g;
        intensity.z = cluster->intensity.b;
        isLeaf = cluster->isLeaf;
        ID = cluster->ID;
        repId = cluster->repId;
        bounds = cluster->bounds;
        memcpy(siblingIDs, cluster->siblingIDs, sizeof(siblingIDs));
    }
    LightCluster(){}
    
    //VplGPU::VPL representativeLight;
    Point repLightHitPoint;
    Normal repLightNormal;

    // This represents the total light contribution of the cluster
    cl_float3 intensity;
    
    bool isLeaf;
    int ID;
    int siblingIDs[ARITY];
    
    BBox bounds;
    
    int repId;
    //Spectrum errorBound;
    //Spectrum estimatedRadiance;
    
};

// The KdTree is used primarily to find the nearest neighbour.
// When the clusters are merged the root node of the kd-tree conatains the
// light tree.
class KdTree {
public:

    KdTree() {
        unsigned long seedBase = (unsigned long)(WallClockTime() / 1000.0);
        
        m_rng = new RandomGenerator(seedBase);
    }
    
    Cluster *Build(std::vector<Cluster *> clusters, int, Cluster*);
    void SetRepresentativeLight(Cluster *cluster);
    int FindSplitIndex(int endIdx);
    
    struct CompareCluster {
        CompareCluster(int a) { axis = a; }
        
        int axis;
        
        bool operator()(const Cluster *c1, const Cluster *c2) const
        {
            return (c1->bounds.pMin[axis] == c2->bounds.pMin[axis]) ? (c1 < c2) :
            (c1->bounds.pMin[axis] < c2->bounds.pMin[axis]);
        }
    };
    
    friend class LightTree;
    
private:
    
    RandomGenerator *m_rng;
    static const int m_axises = 3;
};

class LightTree {
public:
    
    LightTree(vector<VPL> &vpls);
    
    void BuildLightTree();
    
    KdTree *m_kdTree;
    std::vector<Cluster *> m_clusters;
    unsigned int m_lightTreeSize;
    Cluster *m_lightTree;
    LightCluster *m_lightTreeFlat;
    Cluster **m_lightTreeFlatCluster;
};

#endif
