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

#include "rendersession.h"

#include "pathcpu.h"
#include "pathgpu.h"
#include "vplcpu.h"
#include "reconstructioncutscpu.h"
#include "vplgpu.h"
#include "lightcutsgpu.h"
#include "reconstructioncutsgpu.h"

#include "pointcloud.h"

RenderSession::RenderSession(const string& fileName)
{
    m_config.LoadFile(fileName);
}

void RenderSession::Init()
{
    const unsigned int openclPlatformIndex = m_config.GetInt("opencl.platform.index");
    
    GetOpenCLDevices(openclPlatformIndex);
    
    // Create scene
    m_scene = new Scene(m_config, ACCEL_BVH);
    
    const unsigned int width = m_config.GetInt("image.width");
	const unsigned int height = m_config.GetInt("image.height");
    const string oclDeviceConfig = m_config.GetString("opencl.devices.select");
    const unsigned int forceOclWorkSize = m_config.GetInt("opencl.workgroup.size");
    
    m_film = new Film(width, height);
    
    m_scene->m_camera->m_fieldOfView = m_config.GetFloat("camera.fieldofview");
    m_scene->m_camera->Update(m_film->GetWidth(), m_film->GetHeight());
    
    // Generate point cloud
    //PointCloud pointCloud(m_scene, m_film);
    
    // Device info
    bool haveSelectionString = (oclDeviceConfig != "-"  && oclDeviceConfig.length() > 0) ;
    if (haveSelectionString && (m_oclDevices.size() != oclDeviceConfig.length())) {
        stringstream ss;
        ss << "OpenCL device selection string has the wrong length, must be " <<
        m_oclDevices.size() << " instead of " << oclDeviceConfig.length();
        throw runtime_error(ss.str().c_str());
    }

    for (size_t i = 0; i < m_oclDevices.size(); ++i) {
    
        // For now we only want to run one thread!!! that is why we break from
        // the loop
        if (haveSelectionString) {
            if (oclDeviceConfig.at(i) == '1') {
                if (m_oclDevices[i]->GetType() == DEVICE_TYPE_OPENCL_CPU)
                {
                    //RT_LOG("[Selected OpenCL device: " << m_oclDevices[i]->GetName() << "]");
                    m_oclDevices[i]->SetForceWorkGroupSize(forceOclWorkSize);
                    m_oclDevices[i]->SetData(m_scene);
                    m_selectedDevices.push_back(m_oclDevices[i]);
                    break;
                }
                else if(m_oclDevices[i]->GetType() == DEVICE_TYPE_OPENCL_GPU)
                {
                    //RT_LOG("[Selected OpenCL device: " << m_oclDevices[i]->GetName() << "]");
                    m_oclDevices[i]->SetForceWorkGroupSize(forceOclWorkSize);
                    m_oclDevices[i]->SetData(m_scene);
                    m_selectedDevices.push_back(m_oclDevices[i]);
                    break;
                }
                else if(m_oclDevices[i]->GetType() == DEVICE_TYPE_OPENCL_ACCELERATOR)
                {
                    //RT_LOG("[Selected OpenCL device: " << m_oclDevices[i]->GetName() << "]");
                    m_oclDevices[i]->SetForceWorkGroupSize(forceOclWorkSize);
                    m_oclDevices[i]->SetData(m_scene);
                    m_selectedDevices.push_back(m_oclDevices[i]);
                    break;
                }

               
            }
        } else {
            
            // only use CPU
            if(m_oclDevices[i]->GetType() == DEVICE_TYPE_OPENCL_CPU) {
                //RT_LOG("[Selected OpenCL device: " << m_oclDevices[i]->GetName() << "]");
                m_oclDevices[i]->SetForceWorkGroupSize(forceOclWorkSize);
                m_oclDevices[i]->SetData(m_scene);
                m_selectedDevices.push_back(m_oclDevices[i]);
                break;
            }
        }

    }
    
    const int renderEngineType = m_config.GetInt("renderengine.type");
    switch(renderEngineType) {
        case 0:
        {
            m_renderEngine = new PathCPURenderEngine(m_scene, m_film, &m_filmMutex,
                                                    m_config);
            break;
        }
        case 1:
        {
            
            m_renderEngine = new VplCpuRenderEngine(m_scene, m_film, &m_filmMutex,
                                                    m_config);
            break;
        }
        case 2:
        {
            // Lightcuts mode
            m_renderEngine = new VplCpuRenderEngine(m_scene, m_film, &m_filmMutex,
                                                   m_config, true);
            break;
        }
        case 3:
        {
            m_renderEngine = new ReconstructionCutsCpuRenderEngine(m_scene, m_film,
                                                                   &m_filmMutex,
                                                                   m_config);
            break;
        }
        case 4:
        {
            m_renderEngine = new PathGPURenderEngine(m_scene, m_film, &m_filmMutex,
                                                     m_selectedDevices, m_config);
            
            break;
        }
        case 5:
        {

            m_renderEngine = new VPLGPURenderEngine(m_scene, m_film, &m_filmMutex,
                                                    m_selectedDevices, m_config);
            break;

        }
        case 6:
        {
            m_renderEngine = new LightCutsGpuRenderEngine(m_scene, m_film,
                                                          &m_filmMutex,
                                                          m_selectedDevices,
                                                          m_config);
            break;
        }
        case 7:
        {
            m_renderEngine = new ReconstructionCutsGpuRenderEngine(m_scene,
                                                                   m_film,
                                                                   &m_filmMutex,
                                                                   m_selectedDevices,
                                                                   m_config);
            break;
        }

        default:
            assert (false);
        
    }
    
    m_renderEngine->Start();
    
    m_sessionInitialized = true;
}

void RenderSession::GetOpenCLDevices(const int oclPlatformIndex)
{

    cl::Platform oclPlatform;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    for(size_t i = 0; i  < platforms.size(); ++i) {
		std::string plVendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
        RT_LOG("OpenCL vendor platform " << i << ": " << plVendor.c_str());
    }
    
    if(oclPlatformIndex < platforms.size() && oclPlatformIndex >= 0) {
        oclPlatform = platforms[oclPlatformIndex];
    }
    else {
        throw std::runtime_error("Unable to find OpenCL vendor platform");
    }
    
    // Get available devices on platform
    std::vector<cl::Device> oclDevices;
    oclPlatform.getDevices(CL_DEVICE_TYPE_ALL, &oclDevices);
    for (size_t i = 0; i < oclDevices.size(); ++i) {
        
        RT_LOG("OpenCL Device " << i);
        
        OpenCLIntersectionDevice *device = new OpenCLIntersectionDevice(oclDevices[i]);
        m_oclDevices.push_back(device);
        device->PrintInfo();
        
    }

}

void RenderSession::SaveFilmImage() {
    
    boost::unique_lock<boost::mutex> lock(m_filmMutex);
    
    const string fileName = m_config.GetString("image.filename");
    m_film->UpdateScreenBuffer();
    m_film->SaveImpl(fileName);
}

RenderSession::~RenderSession()
{
    
    m_renderEngine->Stop();
    
    delete m_renderEngine;
    delete m_scene;
    delete m_film;
    
    for (size_t i = 0; i < m_oclDevices.size(); ++i) {
        delete m_oclDevices[i];
    }
}
