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

#ifndef RENDERSESSION_H
#define RENDERSESSION_H

#include "openclintersectiondevice.h"
#include "film.h"
#include "scene.h"

#include "renderengine.h"

using namespace std;

class RenderSession {
public:
    
    RenderSession(const string& fileName);
    ~RenderSession();
    
    void Init();
    
    const Properties &GetProperties() const { return m_config; }
    
    void GetOpenCLDevices(const int openclPlatformIndex);
    std::vector<OpenCLIntersectionDevice *> GetSelectedDevices() { return m_selectedDevices; }
    const RenderEngine *GetRenderEngine() const { return m_renderEngine; }
    const bool GetSessionInitialized() const { return m_sessionInitialized;}
    
    void SaveFilmImage();
    
    Film *m_film;
    boost::mutex m_filmMutex;
    
    Properties m_config;
    
private:
    
    Scene *m_scene;
    
    std::vector<OpenCLIntersectionDevice *> m_oclDevices;
    std::vector<OpenCLIntersectionDevice *> m_selectedDevices;
    RenderEngine *m_renderEngine;
    
    bool m_sessionInitialized;
};

#endif
