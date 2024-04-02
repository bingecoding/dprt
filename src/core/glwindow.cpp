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

#if !defined(DISABLE_OPENGL)

#if defined(WIN32)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "glwindow.h"

#include "pathgpu.h"
#include "vplgpu.h"
#include "lightcutsgpu.h"
#include "reconstructioncutsgpu.h"

const RenderSession* session;

GLWindow::GLWindow(const RenderSession* renderSession)
{
    // Check if we have session
    if(renderSession == NULL) {
        throw runtime_error("GLWindow (No session has been created)");
    }
    
    session = renderSession;
    // need to change this
    m_width = session->m_config.GetInt("image.width");
    m_height = session->m_config.GetInt("image.height");

}

void GLWindow::InitGlut(int argc, char *argv[])
{
    glutInit(&argc, argv);
    
    
	glutInitWindowSize(m_width, m_height);
	
    // Center window
	unsigned int scrWidth = glutGet(GLUT_SCREEN_WIDTH);
	unsigned int scrHeight = glutGet(GLUT_SCREEN_HEIGHT);
	if ((scrWidth + 50 < m_width) || (scrHeight + 50 < m_height)) {
		glutInitWindowPosition(0, 0);
    }
	else {
		glutInitWindowPosition((scrWidth - m_width) / 2, (scrHeight - m_height) / 2);
    }
    
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("dprt");
}

void GLWindow::RunGlut()
{
    glutKeyboardFunc(GLWindow::KeyboardFunc);
    glutDisplayFunc(GLWindow::DisplayFunc);
    
    glutTimerFunc(1000, TimerFunc, 0);
    
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, m_width, m_height);
    glLoadIdentity();
    glOrtho(0.f, m_width - 1.f,
            0.f, m_height - 1.f, -1.f, 1.f);
    
    glutMainLoop();
    
}

void GLWindow::DisplayFunc()
{
    session->m_film->UpdateScreenBuffer();
    const float *pixels = session->m_film->GetScreenBuffer();
    
    glRasterPos2i(0, 0);
    
    glDrawPixels(session->m_film->GetWidth(), session->m_film->GetHeight(),
                 GL_RGB, GL_FLOAT, pixels);
    
    glutSwapBuffers();
}

void GLWindow::TimerFunc(int value)
{

    RenderEngineType renderEngineType = session->GetRenderEngine()->GetEngineType();
    switch (renderEngineType) {
        case PATHCPU:
            break;
        case VPLCPU:
            break;
        case LIGHTCUTSCPU:
            break;
        case PATHGPU:
        {
            PathGPURenderEngine *pre = (PathGPURenderEngine *)session->GetRenderEngine();
            pre->UpdateFilm();
            break;
        }
        case VPLGPU:
        {
            VPLGPURenderEngine *vre = (VPLGPURenderEngine *)session->GetRenderEngine();
            vre->UpdateFilm();
            break;
        }
        case LIGHTCUTSGPU:
        {
            LightCutsGpuRenderEngine *lcutsre = (LightCutsGpuRenderEngine *)session->GetRenderEngine();
            lcutsre->UpdateFilm();
            break;
        }
        case RECONSTRUCTIONCUTSGPU:
        {
            ReconstructionCutsGpuRenderEngine *reccuts =
                (ReconstructionCutsGpuRenderEngine *)session->GetRenderEngine();
            reccuts->UpdateFilm();
            break;
        }
        default:
            assert (false);
    }

	glutPostRedisplay();
    
	glutTimerFunc(500, TimerFunc, 0);
    
}


void GLWindow::KeyboardFunc(unsigned char key, int x, int y)
{
	switch (key) {
        case 27: { // Escape key
            
			delete session;
            
			RT_LOG("Done.");
			exit(EXIT_SUCCESS);
			break;
		}
        default:
			break;
    }
    
    GLWindow::DisplayFunc();
}

#endif
