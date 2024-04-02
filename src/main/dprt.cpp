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

#include <fstream>

#include <boost/filesystem.hpp>
#include <FreeImage.h>

#if !defined(DISABLE_OPENGL)
#include "glwindow.h"
#else
#include "rendersession.h"
#endif

#include "pathgpu.h"
#include "lightcutsgpu.h"
#include "reconstructioncutsgpu.h"
#include "vplgpu.h"

using namespace std;

void (*RT_DebugHandler)(const char *msg) = NULL;

void RTDebugHandler(const char *msg) {
    std::cerr << "[RayTracer] " << msg << std::endl;
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
    printf("\n*** ");
    if(fif != FIF_UNKNOWN)
    printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
    
    printf("%s", message);
    printf(" ***\n");
}

int BatchMode(RenderSession *session, double stopTime, unsigned int stopSPP) {
    const double startTime = WallClockTime();
    double lastFilmUpdate = WallClockTime();

    double sampleSec = 0.0;
    char buf[512];
    
    const vector<OpenCLIntersectionDevice *> interscetionDevices = session->GetSelectedDevices();
    for (;;) {
        boost::this_thread::sleep(boost::posix_time::millisec(1000));
        const double now = WallClockTime();
        const double elapsedTime =now - startTime;
        
        const unsigned int engineType = session->GetRenderEngine()->GetEngineType();
        
        unsigned int pass = session->GetRenderEngine()->GetPass();
        unsigned long long samples = session->GetRenderEngine()->GetSamplesCount();
        
        if ((stopTime > 0) && (elapsedTime >= stopTime)){
            sprintf(buf, "[Elapsed time: %3d/%dsec][Passes %4d/%d][Samples %7llu]",
                    int(elapsedTime), int(stopTime), pass, stopSPP, samples);
            cerr << buf << endl;
            break;
        }
        if ((stopSPP > 0) && (pass >= stopSPP)) {
            sprintf(buf, "[Elapsed time: %3d/%dsec][Passes %4d/%d][Samples %7llu]",
                    int(elapsedTime), int(stopTime), pass, stopSPP, samples);
            cerr << buf << endl;
            break;
        }
        
        /*
        // Check if periodic save is enabled
        if (session->NeedPeriodicSave()) {
            if (config->GetRenderEngine()->GetEngineType() == PATHGPU) {
                // I need to update the Film
                PathGPURenderEngine *pre = (PathGPURenderEngine *)config->GetRenderEngine();
                
                pre->UpdateFilm();
            }
            
            // Time to save the image and film
            session->SaveFilmImage();
        }*/
        
        // Print some information about the rendering progress
        double raysSec = 0.0;
        for (size_t i = 0; i < interscetionDevices.size(); ++i)
            raysSec += interscetionDevices[i]->GetPerformance();
        
        
        switch (session->GetRenderEngine()->GetEngineType()) {
            case PATHCPU: {
                //sampleSec = config->film->GetAvgSampleSec();
                //sprintf(buf, "[Elapsed time: %3d/%dsec][Samples %4d/%d][Avg. samples/sec % 4dK][Avg. rays/sec % 4dK on %.1fK tris]",
                //        int(elapsedTime), int(stopTime), pass, stopSPP, int(sampleSec/ 1000.0),
                //        int(raysSec / 1000.0), config->scene->dataSet->GetTotalTriangleCount() / 1000.0);
                break;
            }
            case PATHGPU: {
                PathGPURenderEngine *pre = (PathGPURenderEngine *)session->GetRenderEngine();
                //sampleSec = pre->GetTotalSamplesSec();
                
                sprintf(buf, "[Elapsed time: %3d/%dsec][Samples %4d/%d][Avg. rays/sec % 4dK]",
                        int(elapsedTime), int(stopTime), pass, stopSPP,
                        int(raysSec / 1000.0));
                
                if (WallClockTime() - lastFilmUpdate > 5.0) {
                    pre->UpdateFilm();
                    lastFilmUpdate = WallClockTime();
                }
                break;
            }
            case VPLGPU: {
                VPLGPURenderEngine *pre = (VPLGPURenderEngine *)session->GetRenderEngine();
                //sampleSec = pre->GetTotalSamplesSec();
                
                sprintf(buf, "[Elapsed time: %3d/%dsec][Passes %4d/%d][Samples %7llu]",
                        int(elapsedTime), int(stopTime), pass, stopSPP, pre->GetSamplesCount());
                
                if (WallClockTime() - lastFilmUpdate > 1.0) {
                    pre->UpdateFilm();
                    lastFilmUpdate = WallClockTime();
                }
                break;
            }
            case LIGHTCUTSGPU: {
                LightCutsGpuRenderEngine *pre = (LightCutsGpuRenderEngine *)session->GetRenderEngine();
                //sampleSec = pre->GetTotalSamplesSec();
                
                sprintf(buf, "[Elapsed time: %3d/%dsec][Passes %4d/%d][Samples %7llu]",
                        int(elapsedTime), int(stopTime), pass, stopSPP, pre->GetSamplesCount());
                
                if (WallClockTime() - lastFilmUpdate > 1.0) {
                    pre->UpdateFilm();
                    lastFilmUpdate = WallClockTime();
                }
                break;
            }
            case RECONSTRUCTIONCUTSGPU: {
                ReconstructionCutsGpuRenderEngine *pre = (ReconstructionCutsGpuRenderEngine *)session->GetRenderEngine();
                //sampleSec = pre->GetTotalSamplesSec();
                
                sprintf(buf, "[Elapsed time: %3d/%dsec][Passes %4d/%d][Samples %7llu]",
                        int(elapsedTime), int(stopTime), pass, stopSPP, pre->GetSamplesCount());
                
                if (WallClockTime() - lastFilmUpdate > 1.0) {
                    pre->UpdateFilm();
                    lastFilmUpdate = WallClockTime();
                    /*unsigned long long samplesCount = pre->GetSamplesCount();
                    if(samplesCount == 0) {
                        throw runtime_error("Kernel not updating frame buffer");
                    }*/
                }
                break;
            }
            default:
            assert (false);
        }
        
        cerr << buf << endl;
    }
    
    // Stop the rendering
    //session->StopAllRenderThreads();
    
    // Save the rendered image
    session->SaveFilmImage();
    
    //sprintf(buf, "RayTracer index: %.3f", sampleSec / 1000000.0);
    //cerr << buf << endl;
    
    delete session;
    cerr << "Done." << endl;
    
    return EXIT_SUCCESS;
}

int main(int argc, char * argv[])
{

    try {
        
        // Comment this line to disable logging
        RT_DebugHandler = ::RTDebugHandler;
        
        boost::filesystem::path full_path(boost::filesystem::current_path());
        RT_LOG("Current path is: " << full_path);

        cerr << "Usage: " << argv[0] << " [options] [configuration file]" << endl <<
        " -o [configuration file]" << endl <<
        " -h <display this help and exit>" << endl;
        
        // Initialize FreeImage Library
        FreeImage_Initialise(TRUE);
        FreeImage_SetOutputMessage(FreeImageErrorHandler);
        
        RenderSession *session = NULL;
        
        for(int i = 1; i < argc; i++) {
            if (argv[i][0] == '-') {
                // I should check for out of range array index...
                
                if (argv[i][1] == 'h') exit(EXIT_SUCCESS);
                
                else if (argv[i][1] == 'o') {
                    if(session)
                        throw runtime_error("Used multiple configuration files");
                    
                    session = new RenderSession(argv[++i]);
                }
            }
        }

        if(session == NULL){
           session = new RenderSession("scenes/cornell/render.cfg");
        }
        
        bool batchMode = false;
        const unsigned int halttime = session->m_config.GetInt("batch.halttime");
        const unsigned int haltspp = session->m_config.GetInt("batch.haltspp");
        if ((halttime > 0) || (haltspp > 0)) {
            batchMode = true;
        } else {
            batchMode = false;
        }
        
        if(batchMode) {
            session->Init();
            BatchMode(session, halttime, haltspp);
        }
        #if !defined(DISABLE_OPENGL)
        else {
            GLWindow window(session);
            window.InitGlut(argc, argv);
            
            session->Init();
            
            window.RunGlut();
        }
        #endif

    }
    catch (cl::Error err) {
        cerr << "OpenCL ERROR: " << err.what() << "(" << err.err() << ")" << endl;
        return EXIT_FAILURE;
    }
    catch (std::runtime_error err) {
        cerr << "RUNTIME ERROR: " << err.what() << endl;
        return EXIT_FAILURE;
    }
    catch(std::exception err) {
        cerr << "ERROR: " << err.what() << endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
    
}

