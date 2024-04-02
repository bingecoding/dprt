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

#include "utils/pointcloud.h"

PointCloud::PointCloud(Scene *scene, Film *film) {
    
    m_film = film;
    m_scene = scene;
    
    m_sceneBox = scene->m_bbox;
    const TriangleMesh *preprocessedMesh = scene->m_accel->GetPreprocessedMesh();
    m_triangles = preprocessedMesh->GetTriangles();
    
    const unsigned int normalsCount = scene->GetTotalVertexCount();
    m_normals = new Normal[normalsCount];
    unsigned int nIndex = 0;
    for (unsigned int i = 0; i < scene->m_objectMeshes.size(); ++i) {
        TriangleMesh *mesh = scene->m_objectMeshes[i];
        
        for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
            m_normals[nIndex++] = mesh->GetNormal(j);
    }
    
    m_xLength = fabs(m_sceneBox.pMax.x - m_sceneBox.pMin.x);
    m_yLength = fabs(m_sceneBox.pMax.y - m_sceneBox.pMin.y);
    m_zLength = fabs(m_sceneBox.pMax.z - m_sceneBox.pMin.z);
    
    m_xOffset = fabs(m_sceneBox.pMin.x);
    m_yOffset = fabs(m_sceneBox.pMin.y);
    m_zOffset = fabs(m_sceneBox.pMin.z);
    
    m_xResolution = floor(m_xLength) * 15;
    m_yResolution = floor(m_yLength) * 15;
    m_zResolution = floor(m_zLength) * 15;
    
    
    m_points = new int[m_xResolution * m_yResolution * m_zResolution];
    
    for(int i=0; i < m_xResolution; i++)
        for(int j=0; j < m_yResolution; j++)
            for (int k=0; k < m_zResolution; k++) {
                int index = i + j * m_xResolution + k * m_xResolution * m_yResolution;
                m_points[index] = 1;
            }
    
    unsigned long seedBase = (unsigned long)(WallClockTime() / 1000.0);
    
    m_rndGen = new RandomGenerator(seedBase);
    
    GeneratePointCloud();
    
    WritePointCloud();
    /*
    // Intersection code - sample point but for which scene?
    Point p0(-3.f, 1.f, 3.f);
    Point p1(-4.f,4.878, 2.397);
    Point o1,o2;
    
    GetOffset(p0, &o1);
    GetOffset(p1, &o2);
    float d = Distance(o2, o1);
    Vector dir(o2 - o1);
    dir = Normalize(dir);
    
    int step = round(d);
    for(int i=0; i < step; i++) {
        Point traverse;
        traverse = o1 + dir*i;
        
        int xp = floor(traverse.x);
        int yp = floor(traverse.y);
        int zp = floor(traverse.z);
        
        int index = xp + yp * m_xResolution + zp * m_xResolution * m_yResolution;
        if(m_points[index] == 1) {
            std::cout << "hit point coordinates: " << xp << " "<<yp <<" " << zp<< std::endl;
        }
    }
    */
}

void PointCloud::GeneratePointCloud()
{
    
    const unsigned int width = m_film->GetWidth();
    const unsigned int height = m_film->GetHeight();
    
    const unsigned int pixelCount = width * height;
    
    for (unsigned int i = 0; i < pixelCount; ++i) {
    for(int k= 0; k < 10; k++) {
        Ray ray;
        RayHit rayHit;
        
        // Generate rays from camera
        const unsigned int x = i % width;
        const unsigned int y = i / width;
        const float scrX = x + m_rndGen->floatValue() - 0.5f;
        const float scrY = y + m_rndGen->floatValue() - 0.5f;
        m_scene->m_camera->GenerateRay(scrX, scrY, width, height, &ray,
                                     m_rndGen->floatValue(), m_rndGen->floatValue(),
                                     m_rndGen->floatValue());
        
        RayHit restoreRayHit;
        Ray restoreRay;
        
        // Shoot ray into scene from camera
        if(m_scene->m_accel->Intersect(&ray, &rayHit)) {
            
            // Let ray bounce around in the scene
            int j= 0; // 5 is max depth
            while(j++ < 5) {
                
                // Choose new direction for ray
                // Just sample hemisphere
                Vector dir = UniformSampleSphere(m_rndGen->floatValue(), m_rndGen->floatValue());
                
                Triangle tri = m_triangles[rayHit.index];
                Normal N;
                const float b0 = 1.f - rayHit.b1 - rayHit.b2;
                N = b0 * m_normals[tri.v[0]] +
                rayHit.b1 * m_normals[tri.v[1]] +
                rayHit.b2 * m_normals[tri.v[2]];
                Normalize(N);
                
                //flip ray if on wrong side
                float dotN = Dot(dir, N);
                if (dotN < 0.f) {
                    dir *= -1.f;
                }
                
                // save ray information
                restoreRayHit = rayHit;
                restoreRay = ray;
                
                // setup new ray direction
                Point hitPoint;
                hitPoint = ray.o + ray.d * rayHit.t;
                
                ray.o = hitPoint;
                ray.d = dir;
                
                if(m_scene->m_accel->Intersect(&ray, &rayHit)) {
                    
                    hitPoint = ray.o + ray.d * rayHit.t;
                    
                    // Map point to point in cloud
                    // The points along the xyz axises in the
                    // point cloud are always positive,
                    // so we have to adjust that here.
                    Point currOffsetHitPoint;
                    GetOffset(hitPoint, &currOffsetHitPoint);
                    
                    Point prevOffsetHitPoint;
                    Point prevHitPoint = ray.o;
                    GetOffset(prevHitPoint, &prevOffsetHitPoint);
                    
                    float d = Distance(currOffsetHitPoint, prevOffsetHitPoint);
                    Vector dir(currOffsetHitPoint - prevOffsetHitPoint);
                    dir = Normalize(dir);
                    
                    int step = floor(d);
                    for(int i=0; i < step; i++) {
                        
                        Point offsetFillPoint;
                        offsetFillPoint = prevOffsetHitPoint + dir*i;
                        
                        int xp = floor(offsetFillPoint.x);
                        int yp = floor(offsetFillPoint.y);
                        int zp = floor(offsetFillPoint.z);
                        
                        //assert(xp <= m_xResolution && yp <= m_yResolution && zp <= m_zResolution);
                        //assert(xp >= 0 && yp >= 0 && zp >= 0);
                        
                        // Preserve edges of box
                        
                        if(xp <= 0){
                            xp = 1;
                        }
                        if(xp >= m_xResolution - 1){
                            xp = m_xResolution - 2;
                        }
                      
                        if(zp <= 0){
                            zp = 1;
                        }
                        if(zp >= m_zResolution - 1){
                            zp = m_zResolution - 2;
                        }
                        
                        if(yp <= 0){
                            yp = 1;
                        }
                        if(yp >= m_yResolution - 1){
                            yp = m_yResolution - 2;
                        }

                        int index = xp + yp * m_xResolution + zp * m_xResolution * m_yResolution;
                        m_points[index] = 0;
                    }
                    
                }else {
                    //rayHit = restoreRayHit;
                    //ray = restoreRay;
                    break;
                }
            }
            
            
        }
    }
    }
    
}


void PointCloud::GetOffset(const Point &hitPoint, Point *offsetPoint)
{
    
    // The point cloud only has coordinates at positive points in the xyz
    // axises so we shift the points to positive positions.
    if(hitPoint.x < 0.f) {
        offsetPoint->x = m_xOffset - fabs(hitPoint.x);
    }else{
        offsetPoint->x = m_xOffset + hitPoint.x;
    }
    
    if(hitPoint.y < 0.f) {
        offsetPoint->y = m_yOffset - fabs(hitPoint.y);
    }else {
        offsetPoint->y = m_yOffset + hitPoint.y;
    }
    
    if(hitPoint.z < 0.f) {
        offsetPoint->z =  m_zOffset - fabs(hitPoint.z);
    }else {
        offsetPoint->z = m_zOffset + hitPoint.z;
    }
    
    // Now set offset according to resolution of point cloud and size of
    // the scene. The array's last element is the resolution -1.
    float xRes = (float)(m_xResolution -1.f) / m_xLength;
    float yRes = (float)(m_yResolution -1.f) / m_yLength;
    float zRes = (float)(m_zResolution -1.f) / m_zLength;
    
    float xResOff = offsetPoint->x * xRes;
    float yResOff = offsetPoint->y * yRes;
    float zResOff = offsetPoint->z * zRes;
    
    offsetPoint->x = xResOff;
    offsetPoint->y = yResOff;
    offsetPoint->z = zResOff;
    
}

Point PointCloud::GetOffset(const Point &hitPoint)
{
    
    // The point cloud only has coordinates at positive points in the xyz
    // axises so we shift the points to positive positions.
    Point offsetPoint;
    if(hitPoint.x < 0.f) {
        offsetPoint.x = m_xOffset - fabs(hitPoint.x);
    }else{
        offsetPoint.x = m_xOffset + hitPoint.x;
    }
    
    if(hitPoint.y < 0.f) {
        offsetPoint.y = m_yOffset - fabs(hitPoint.y);
    }else {
        offsetPoint.y = m_yOffset + hitPoint.y;
    }
    
    if(hitPoint.z < 0.f) {
        offsetPoint.z =  m_zOffset - fabs(hitPoint.z);
    }else {
        offsetPoint.z = m_zOffset + hitPoint.z;
    }
    
    // Now set offset according to resolution of point cloud and size of
    // the scene. The array's last element is the resolution -1.
    float xRes = (float)(m_xResolution -1.f) / m_xLength;
    float yRes = (float)(m_yResolution -1.f) / m_yLength;
    float zRes = (float)(m_zResolution -1.f) / m_zLength;
    
    float xResOff = offsetPoint.x * xRes;
    float yResOff = offsetPoint.y * yRes;
    float zResOff = offsetPoint.z * zRes;
    
    offsetPoint.x = xResOff;
    offsetPoint.y = yResOff;
    offsetPoint.z = zResOff;
    
    return offsetPoint;
    
}


void PointCloud::WritePointCloud()
{
    std::ofstream myfile;
    myfile.open("../scenes/cornell/sibenikPointCloud.obj");
    
    for(int i=0; i < m_xResolution; i++)
        for(int j=0; j < m_yResolution; j++)
            for (int k=0; k < m_zResolution; k++) {
                int index = i + j * m_xResolution + k * m_xResolution * m_yResolution;
                if(m_points[index] == 1) {
                    myfile << "v " << i << " " << j << " " << k << "\n";
                }
            }
    
     myfile.close();
   
}
