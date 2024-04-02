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

#ifndef INTERSECTIONDEVICE_H
#define INTERSECTIONDEVICE_H

#include <iostream>
#include <string>

#include "raytracer.h"
#include "scene.h"

typedef enum {
    DEVICE_TYPE_NATIVE_THREAD,
	DEVICE_TYPE_OPENCL_CPU,
	DEVICE_TYPE_OPENCL_GPU,
    DEVICE_TYPE_OPENCL_UNKNOWN,
    DEVICE_TYPE_OPENCL_ACCELERATOR,
} DeviceType;

static inline DeviceType GetOCLDeviceType(const cl_int type)
{
    
    switch (static_cast<unsigned int>(type)) {
        case CL_DEVICE_TYPE_CPU:
            return DEVICE_TYPE_OPENCL_CPU;
        case CL_DEVICE_TYPE_GPU:
            return DEVICE_TYPE_OPENCL_GPU;
        case CL_DEVICE_TYPE_ACCELERATOR:
            return DEVICE_TYPE_OPENCL_ACCELERATOR;
        default:
            return DEVICE_TYPE_OPENCL_UNKNOWN;
    }
    
}

static inline std::string GetOCLDeviceType(const DeviceType type)
{
    
    switch (type) {
        case DEVICE_TYPE_NATIVE_THREAD:
            return "Native Thread";
        case DEVICE_TYPE_OPENCL_CPU:
            return "CPU";
        case DEVICE_TYPE_OPENCL_GPU:
            return "GPU";
        case DEVICE_TYPE_OPENCL_ACCELERATOR:
            return "ACCELERATOR";
        default:
            return "UNKNOWN";
    }

}

class IntersectionDevice {
public:
    
    IntersectionDevice(const std::string deviceName,
                    const DeviceType deviceType) :
    m_name(deviceName), m_type(deviceType), m_scene(NULL) { }
	
    virtual ~IntersectionDevice() { }
    
    const std::string &GetName() const { return m_name; }
	const DeviceType GetType() const { return m_type; };
	virtual int GetComputeUnits() const { return 1; }
	virtual unsigned int GetNativeVectorWidthFloat() const { return 4; };
	virtual size_t GetMaxMemory() const { return 0; }
	virtual size_t GetMaxMemoryAllocSize() const { return 0; }
    virtual size_t GetMaxWorkGroupSize() const { return 0; }
    virtual size_t GetMaxLocalMemory() const { return 0; }
    
    virtual void PrintInfo () const
    {
        RT_LOG("OpenCL name: " << GetName());
        RT_LOG("OpenCL type: " << GetOCLDeviceType(GetType()));
        RT_LOG("OpenCL compute units: " << GetComputeUnits());
        RT_LOG("OpenCL memory: " << GetMaxMemory() / (1024 * 1024) << " MB");
        RT_LOG("OpenCL local memory: " << GetMaxLocalMemory() / 1024 << " Kb");
        RT_LOG("OpenCL maximum work-group size: " << GetMaxWorkGroupSize());
    }
    
    const Scene *GetScene() const { return m_scene; }
    
    double GetPerformance() const {
        const double statsTotalRayTime = WallClockTime() - m_statsStartTime;
        return (statsTotalRayTime == 0.0) ? 1.0 : (m_statsTotalRayCount /statsTotalRayTime);
    }
    
    void ResetPerformanceStats() {
        m_statsStartTime = WallClockTime();
        m_statsTotalRayCount = 0;
    }

    
protected:
    
    const Scene *m_scene;
    
    std::string m_name;
	DeviceType m_type;
   
    double m_statsStartTime, m_statsTotalRayCount;
    double  m_statsDeviceIdleTime, m_statsDeviceTotalTime;
};


#endif
