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

#ifndef __RayTracer__pointcloud__
#define __RayTracer__pointcloud__

#include <iostream>

#include "utils/scene.h"
#include "accelerator.h"
#include "film.h"
#include "utils/sampler.h"

class PointCloud {
public:
    
    PointCloud(Scene *scene, Film *film);
    
    const int *GetPointCloud() const  { return m_points; }
    const int GetCloudPoint(int index) const  { return m_points[index]; }
    void GetOffset(const Point &hitPoint, Point *offsetPoint);
    Point GetOffset(const Point &hitPoint);
    
    int m_xResolution, m_yResolution, m_zResolution;
    
    float m_xLength, m_yLength, m_zLength;
    float m_xOffset, m_yOffset, m_zOffset;

private:
    
    void GeneratePointCloud();
   
    void WritePointCloud();
    
    Scene *m_scene;
    Film *m_film;
    BBox m_sceneBox;
    
    const Triangle *m_triangles;
    Normal *m_normals;
    RandomGenerator *m_rndGen;
    
    int *m_points;
    
};

#endif /* defined(__RayTracer__pointcloud__) */
