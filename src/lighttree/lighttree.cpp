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

#include "lighttree/lighttree.h"
#include "montecarlo.h"

#include <queue>

LightTree::LightTree(vector<VPL> &vpls) : m_lightTreeSize(0) {

    for(int i=0; i < vpls.size(); i++) {
        Cluster *cluster = new Cluster();
        cluster->representativeLight = vpls[i];
        cluster->bounds = BBox(vpls[i].hitPoint);
        cluster->intensity = vpls[i].contrib;
        cluster->isLeaf = true;
        m_clusters.push_back(cluster);
    }
    m_kdTree = new KdTree();
}

void LightTree::BuildLightTree()
{
    
    Cluster *lightTree = m_kdTree->Build(m_clusters, 0, NULL);
    m_kdTree->SetRepresentativeLight(lightTree);

    // Assign -1 to IDs for empty children and find size of tree.
    std::queue<Cluster *> queueLightTree;
    queueLightTree.push(lightTree);
    m_lightTreeSize = 1;
    while(!queueLightTree.empty()) {
        Cluster* cluster = queueLightTree.front();
        queueLightTree.pop();
        
        for(int i=0; i < ARITY; i++) {
            if(cluster->siblings[i] != NULL) {
                m_lightTreeSize++;
                cluster->siblingIDs[i] = cluster->ID * 2 + i + 1;
                queueLightTree.push(cluster->siblings[i]);
            }else {
                cluster->siblingIDs[i] = -1;
            }
        }
    }
    
    m_lightTree = lightTree;
    
    // Put light tree into heap data structure, because the tree is balanced
    // there is no wasted space in memory.
    m_lightTreeFlat = new LightCluster[m_lightTreeSize];
    m_lightTreeFlatCluster = new Cluster*[m_lightTreeSize];
    std::cout << "Heap size of light tree: "<< sizeof(LightCluster) * m_lightTreeSize/1024 << " kB" << std::endl;
    queueLightTree.push(lightTree);
    while(!queueLightTree.empty()) {
        Cluster* cluster = queueLightTree.front();
        queueLightTree.pop();
        
        if(cluster != NULL) {
            m_lightTreeFlat[cluster->ID] = cluster;
            m_lightTreeFlatCluster[cluster->ID] = cluster;
        }
        
        for(int i=0; i < ARITY; i++) {
            if(cluster->siblings[i] != NULL) {
                queueLightTree.push(cluster->siblings[i]);
            }
        }
        
    }
    
}

int KdTree::FindSplitIndex(int endIdx)
{
    int N = endIdx;
    int M = pow(2, floor(log(N)/log(2)));
    int R = 2*M - N;
    int X = Max(R - M/2.f, 0.f);
    int offset = M - X;
    return offset;
}


Cluster *KdTree::Build(std::vector<Cluster *> clusters, int nodeIdx, Cluster* parent)
{
    Cluster *cluster = new Cluster();
    if(clusters.size() == 1) {
        cluster->representativeLight = clusters[0]->representativeLight;
        cluster->repLight = true;
        cluster->bounds = clusters[0]->bounds;
        cluster->intensity = clusters[0]->intensity;
        cluster->isLeaf = true;
        cluster->ID = nodeIdx;
        return cluster;
    }
    
    
    cluster->ID = nodeIdx;
    //std::vector<float> lightPower;
    //int nLights = 0;
    //bool parentRepresentativeLight = false;
    for(int i=0; i < clusters.size(); i++) {
        cluster->bounds = Union(cluster->bounds, clusters[i]->bounds);
        cluster->intensity += clusters[i]->intensity;
        //lightPower.push_back(clusters[i]->intensity.Y());
        //nLights++;
        //if(parent != NULL && (parent->representativeLight.hitPoint == clusters[i]->representativeLight.hitPoint)) {
        //    parentRepresentativeLight = true;
        //}
    }
    /*
    if(parentRepresentativeLight) {
        cluster->representativeLight = parent->representativeLight;
    }
    else {
     
        Distribution1D *lightDistribution = new Distribution1D(&lightPower[0],
        nLights);
        // Choose one of the clusters randomly based on its relative intensity
        float lightPdf;
        int ln = lightDistribution->SampleDiscrete(m_rng->floatValue(),
        &lightPdf);

        cluster->representativeLight = clusters[ln]->representativeLight;
    }
    */
    int splitAxis = cluster->bounds.MaximumExtent();
    int splitIdx = FindSplitIndex(clusters.size());
    
    std::vector<Cluster *> clustersCopy(clusters);
    std::nth_element(clustersCopy.begin(), clustersCopy.begin() + splitIdx,
                     clustersCopy.end() , CompareCluster(splitAxis));

    std::vector<Cluster *> cluster1(clustersCopy.begin(), clustersCopy.begin() + splitIdx);
    std::vector<Cluster *> cluster2(clustersCopy.begin()+splitIdx, clustersCopy.end());
    
    cluster->siblings[0] = Build(cluster1, (nodeIdx*2)+1, cluster);
    cluster->siblings[1] = Build(cluster2, (nodeIdx*2)+2, cluster);
    
    return cluster;
}

// Bottom-up approach for selecting representative light.
// Top-down approach quicker and qualtiy seems just as good!
void KdTree::SetRepresentativeLight(Cluster *cluster){
    
    if(cluster->repLight == true || cluster == NULL) {
        return;
    }
    
    SetRepresentativeLight(cluster->siblings[0]);
    SetRepresentativeLight(cluster->siblings[1]);
    
    std::vector<float> lightPower;
    int nLights = 0;
    for (int i = 0; i < ARITY; ++i) {
        if(cluster->siblings[i] != NULL) {
            lightPower.push_back(cluster->siblings[i]->intensity.Y());
            nLights++;
        }
    }
    Distribution1D *lightDistribution = new Distribution1D(&lightPower[0],
                                                           nLights);
    
    // Choose one of the clusters randomly based on its relative intensity
    float lightPdf;
    int ln = lightDistribution->SampleDiscrete(m_rng->floatValue(),
                                               &lightPdf);
    
    cluster->representativeLight = cluster->siblings[ln]->representativeLight;
    cluster->repId = cluster->siblings[ln]->ID;
}
