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

#ifndef GLWINDOW_H
#define GLWINDOW_H

#if !defined(DISABLE_OPENGL)

#include "rendersession.h"

class GLWindow {
public:
    
    GLWindow(const RenderSession* session);
    
    void InitGlut(int argc, char *argv[]);
    void RunGlut();
    
private:
        
    static void DisplayFunc();
    static void KeyboardFunc(unsigned char key, int x, int y);
    static void TimerFunc(int value);
    
    int m_width;
    int m_height;
    
};

#endif

#endif 
