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

#ifndef FILM_H
#define FILM_H

#include "raytracer.h"
#include "film/buffer.h"

static const unsigned int GammaTableSize = 1024;

class Film {
public:
    
    Film(const unsigned int width, const unsigned int height);
    ~Film();
    
    void Init(const unsigned int w, const unsigned int h);
    void InitGammaTable(const float gamma  = 2.2f);
    
    unsigned int GetWidth() { return m_width; }
	unsigned int GetHeight() { return m_height; }
    
    SampleBuffer *GetFreeSampleBuffer();
    void FreeSampleBuffer(SampleBuffer *sampleBuffer);
    void Reset();
    
    void SplatSampleBuffer(const bool preview, SampleBuffer *sampleBuffer);
    void SplatRadiance(const Spectrum radiance, const unsigned int x,
                       const unsigned int y, const float weight = 1.f);
    
    static size_t SampleBufferSize;
    
    void UpdateScreenBuffer();
    
    void SaveImpl(const std::string &fileName);
    
    const float *GetScreenBuffer() const;
    
    float Radiance2PixelFloat(const float x) const;
    float m_gammaTable[GammaTableSize];
    SampleBuffer *m_sampleBuffer;
    
private:

    unsigned int m_width, m_height;
    unsigned int m_pixelCount;
    
    SampleFrameBuffer *m_sampleFrameBuffer;
    FrameBuffer *m_frameBuffer;
   
    
    boost::mutex m_splatMutex;
    
};

#endif
