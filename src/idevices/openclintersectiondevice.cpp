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

#include "idevices/openclintersectiondevice.h"

size_t OpenCLIntersectionDevice::RayBufferSize = OPENCL_RAYBUFFER_SIZE;

OpenCLIntersectionDevice::OpenCLIntersectionDevice(cl::Device &device) :
IntersectionDevice(device.getInfo<CL_DEVICE_NAME>().c_str(),
                   GetOCLDeviceType(device.getInfo<CL_DEVICE_TYPE>())),
m_oclDevice(device),
m_oclContext(NULL),
m_usedMemory(0),
m_forceWorkGroupSize(0),
m_computeUnits(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()),
m_maxMemory(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()),
m_maxWorkGroupSize(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()),
m_maxLocalMemory(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>())
{
    m_bvhKernel = NULL;
	
	m_oclQueue = NULL;
    
	m_bvhBuff = NULL;
	m_vertsBuff = NULL;
	m_trisBuff = NULL;
	m_bvhBuff = NULL;
    
    
	// Allocate the queue for this device
    GetOpenCLContext();
	m_oclQueue = new cl::CommandQueue(*m_oclContext, m_oclDevice);
}

OpenCLIntersectionDevice::~OpenCLIntersectionDevice()
{

	FreeDataSetBuffers();
    
	delete m_bvhKernel;
    delete m_oclQueue;
}

cl::Context &OpenCLIntersectionDevice::GetOpenCLContext() const
{
    if (!m_oclContext) {
        // Allocate a context with the selected device
        VECTOR_CLASS<cl::Device> devices;
        devices.push_back(m_oclDevice);
        cl::Platform platform = m_oclDevice.getInfo<CL_DEVICE_PLATFORM>();
        
        cl_context_properties cps[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0
        };
        
        m_oclContext = new cl::Context(devices, cps);
        
    }
    
    return *m_oclContext;
}

void OpenCLIntersectionDevice::EnqueueTraceRayBuffer(cl::Buffer &rBuff,
                                                     cl::Buffer &hBuff,
                                                     const unsigned int rayCount)
{
	switch (m_scene->GetAcceleratorType()) {
		case ACCEL_BVH: {
			m_bvhKernel->setArg(0, rBuff);
			m_bvhKernel->setArg(1, hBuff);
			m_bvhKernel->setArg(7, rayCount);
			m_oclQueue->enqueueNDRangeKernel(*m_bvhKernel, cl::NullRange,
                                             cl::NDRange(rayCount),
                                             cl::NDRange(m_bvhWorkGroupSize));
			break;
		}
		default:
			assert (false);
	}
    
	m_statsTotalRayCount += rayCount;
}

void OpenCLIntersectionDevice::FreeDataSetBuffers() {
	// Check if I have to free something from previous DataSet
	if (m_scene) {
		m_usedMemory -= m_raysBuff->getInfo<CL_MEM_SIZE>();
		delete m_raysBuff;
		m_usedMemory -= m_hitsBuff->getInfo<CL_MEM_SIZE>();
		delete m_hitsBuff;
        
		if (m_bvhBuff) {
			m_usedMemory -= m_vertsBuff->getInfo<CL_MEM_SIZE>();
			delete m_vertsBuff;
			m_usedMemory -= m_trisBuff->getInfo<CL_MEM_SIZE>();
			delete m_trisBuff;
			m_usedMemory -= m_bvhBuff->getInfo<CL_MEM_SIZE>();
			delete m_bvhBuff;
		}
    }
}

void OpenCLIntersectionDevice::SetData(const Scene *data)
{
	
    FreeDataSetBuffers();
    
    m_scene = data;
    
    cl::Context &oclContext = GetOpenCLContext();
    
	// Allocate OpenCL buffers
	RT_LOG("[OpenCL device::" << m_name <<
           "] Ray buffer size: " << (sizeof(Ray) * RayBufferSize / 1024) <<
           "Kbytes");
    
    m_raysBuff = new cl::Buffer(oclContext,
                                CL_MEM_READ_ONLY,
                                sizeof(Ray) * RayBufferSize);
	m_usedMemory += m_raysBuff->getInfo<CL_MEM_SIZE>();
    
	RT_LOG("[OpenCL device::" << m_name <<
           "] Ray hits buffer size: " <<
           (sizeof(RayHit) * RayBufferSize / 1024) << "Kbytes");
	m_hitsBuff = new cl::Buffer(oclContext,
                                CL_MEM_WRITE_ONLY,
                                sizeof(RayHit) * RayBufferSize);
	m_usedMemory += m_hitsBuff->getInfo<CL_MEM_SIZE>();
    
    cl::Device &oclDevice = m_oclDevice;
	switch (m_scene->GetAcceleratorType()) {
		case ACCEL_BVH: {
			//------------------------------------------------------------------
			// BVH kernel
			//------------------------------------------------------------------
            
			{
                std::vector<std::string> files;
                files.push_back("kernels/bvh_kernel.cl");
                cl::Program program = getProgram(oclContext, files);
                try {
                    std::vector<cl::Device> buildDevice;
					buildDevice.push_back(oclDevice);
					program.build(buildDevice, "-I.");
				} catch (cl::Error err) {
					std::string strError = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(oclDevice);
					RT_LOG("[OpenCL device::" << m_name <<
                           "] BVH compilation error:\n" << strError.c_str());
                    
					throw err;
				}
                
                m_bvhKernel = new cl::Kernel(program, "Intersect");
                m_bvhWorkGroupSize = m_forceWorkGroupSize;
                RT_LOG("[OpenCL device::" << m_name <<
                       "] Forced work group size: " << m_bvhWorkGroupSize);
                
            }
            
            RT_LOG("[OpenCL device::" << m_name << "] Vertices buffer size: " <<
                   (sizeof(Point) * m_scene->GetTotalVertexCount() / 1024) << "Kbytes");
			const BVHAccel *bvh = (BVHAccel *)m_scene->m_accel;
			m_vertsBuff = new cl::Buffer(oclContext,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(Point) * m_scene->GetTotalVertexCount(),
                                         bvh->m_preprocessedMesh->GetVertices());
            m_usedMemory += m_vertsBuff->getInfo<CL_MEM_SIZE>();
            
            RT_LOG("[OpenCL device::" << m_name <<
                   "] Triangle indices buffer size: " <<
                   (sizeof(Triangle) * m_scene->GetTotalTriangleCount() / 1024) << "Kbytes");
			m_trisBuff = new cl::Buffer(oclContext,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(Triangle) * m_scene->GetTotalTriangleCount(),
                                        bvh->m_preprocessedMesh->GetTriangles());
			m_usedMemory += m_trisBuff->getInfo<CL_MEM_SIZE>();
            
            RT_LOG("[OpenCL device::" << m_name << "] BVH buffer size: " <<
                   (sizeof(BVHAccelArrayNode) * bvh->m_nNodes / 1024) << "Kbytes");
			m_bvhBuff = new cl::Buffer(oclContext,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(BVHAccelArrayNode) * bvh->m_nNodes,
                                       bvh->m_bvhTree);
			m_usedMemory += m_bvhBuff->getInfo<CL_MEM_SIZE>();
            
			// Set arguments
			m_bvhKernel->setArg(2, *m_vertsBuff);
			m_bvhKernel->setArg(3, *m_trisBuff);
			m_bvhKernel->setArg(4, m_scene->GetTotalTriangleCount());
			m_bvhKernel->setArg(5, bvh->m_nNodes);
			m_bvhKernel->setArg(6, *m_bvhBuff);
            
            break;
        }
        default:
			assert (false);
    }
    
}
