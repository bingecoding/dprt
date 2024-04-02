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

#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include "trianglemesh.h"

typedef enum {
    ACCEL_BVH
} AcceleratorType;

class Accelerator {
    
public:
    
    Accelerator() {}
    virtual ~Accelerator() {}
    
    virtual AcceleratorType GetType() const = 0;
    
    virtual void Init(const std::deque<TriangleMesh *> &meshes, const unsigned int totalVertxCount,
                      const unsigned int totalTriangleCount) = 0;
    
    virtual const TriangleMeshID GetMeshID(const unsigned int index) const = 0;
    virtual const TriangleMeshID *GetMeshIDTable() const = 0;
    virtual const TriangleID GetMeshTriangleID(const unsigned int index) const = 0;
    virtual const TriangleID *GetMeshTriangleIDTable() const = 0;
    virtual const TriangleMesh *GetPreprocessedMesh() const = 0;
    
    virtual bool Intersect(const Ray *ray, RayHit *hit) const = 0;
    virtual bool IntersectP(const Ray *ray) const = 0;
    
};


#endif
