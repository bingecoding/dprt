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

#ifndef OPENCLDEVICE_H
#define OPENCLDEVICE_H

#include "idevices/intersectiondevice.h"

#define OPENCL_RAYBUFFER_SIZE 65536

class OpenCLIntersectionDevice : public IntersectionDevice {
public:
    
    OpenCLIntersectionDevice(cl::Device &device);
    ~OpenCLIntersectionDevice();
    
    int GetComputeUnits() const { return m_computeUnits; }
    size_t GetMaxMemory() const { return m_maxMemory; }
    size_t GetMaxLocalMemory() const { return m_maxLocalMemory; }
    size_t GetMaxWorkGroupSize() const { return m_maxWorkGroupSize; }
    
    cl::Context &GetOpenCLContext() const;
    cl::Device &GetOpenCLDevice() const { return m_oclDevice; }

    cl::CommandQueue &GetOpenCLQueue() { return *m_oclQueue; }

    cl::Buffer &GetOpenCLBVH() { return *m_bvhBuff; }
    cl::Buffer &GetOpenCLVerts() { return *m_vertsBuff; }
    
    unsigned int GetForceWorkGroupSize() const { return m_forceWorkGroupSize; }
    void SetForceWorkGroupSize(const unsigned int size) const
    {
        m_forceWorkGroupSize = size;
    }
    
    void EnqueueTraceRayBuffer(cl::Buffer &rBuff, cl::Buffer &hBuff,
                               const unsigned int rayCount);
    void FreeDataSetBuffers();
    void SetData(const Scene *data);
    
    void AllocMemory(size_t s) const { m_usedMemory += s; }
    void FreeMemory(size_t s) const { m_usedMemory -= s; }
    
    static size_t RayBufferSize;
    
private:
    
    int m_computeUnits;
    size_t m_maxMemory;
    size_t m_maxLocalMemory;
    size_t m_maxWorkGroupSize;
    
    mutable size_t m_usedMemory;
    mutable unsigned int m_forceWorkGroupSize;
    
    mutable cl::Device m_oclDevice;
    
    mutable cl::Context *m_oclContext;
    cl::CommandQueue *m_oclQueue;
    
    cl::Kernel *m_bvhKernel;
	size_t m_bvhWorkGroupSize;
	cl::Buffer *m_vertsBuff;
	cl::Buffer *m_trisBuff;
	cl::Buffer *m_bvhBuff;
    
	cl::Buffer *m_raysBuff;
	cl::Buffer *m_hitsBuff;

};


#endif
