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

#include <FreeImage.h>

#include "film/film.h"

size_t Film::SampleBufferSize = 4096;

Film::Film(const unsigned int width, const unsigned int height)
{
    m_sampleFrameBuffer = NULL;
    m_frameBuffer = NULL;
    m_sampleBuffer = NULL;
    
    InitGammaTable();
    
    Init(width, height);
}

Film::~Film()
{
    delete m_sampleBuffer;
    delete m_sampleFrameBuffer;
    delete m_frameBuffer;
}

const float *Film::GetScreenBuffer() const
{
    return (const float *)m_frameBuffer->GetPixels();
}

void Film::Init(const unsigned int w, const unsigned int h)
{
    m_width = w;
    m_height = h;
    RT_LOG("Film size " << m_width << "x" << m_height);
    m_pixelCount = w * h;
    
    delete m_sampleFrameBuffer;
    delete m_frameBuffer;
    
    m_sampleFrameBuffer = new SampleFrameBuffer(m_width, m_height);
    m_sampleFrameBuffer->Clear();
    
    m_frameBuffer = new FrameBuffer(m_width, m_height);
    m_frameBuffer->Clear();
    
}

void Film::InitGammaTable(const float gamma) {
    float x = 0.f;
    const float dx = 1.f / GammaTableSize;
    for (unsigned int i = 0; i < GammaTableSize; ++i, x += dx)
        m_gammaTable[i] = powf(Clamp(x, 0.f, 1.f), 1.f / gamma);
}

SampleBuffer *Film::GetFreeSampleBuffer() {
    
    boost::mutex::scoped_lock lock(m_splatMutex);
    
    if (m_sampleBuffer == NULL) {
        // Need to allocate a new buffer
        m_sampleBuffer = new SampleBuffer(SampleBufferSize);
    }
    // This is important
    m_sampleBuffer->Reset();
    return m_sampleBuffer;
}

void Film::Reset()
{
    m_sampleFrameBuffer->Clear();
}

void Film::SplatSampleBuffer(const bool preview, SampleBuffer *sampleBuffer)
{
    boost::mutex::scoped_lock lock(m_splatMutex);
	
    // could apply filters here 
    const SampleBufferElem *sbe = sampleBuffer->GetSampleBuffer();
    for (unsigned int i = 0; i < sampleBuffer->GetSampleCount(); ++i) {
        const SampleBufferElem *sampleElem = &sbe[i];
        const int x = (int)sampleElem->screenX;
        const int y = (int)sampleElem->screenY;
        
        SplatRadiance(sampleElem->radiance, x, y, 1.f);
    }

}

void Film::SplatRadiance(const Spectrum radiance, const unsigned int x,
                   const unsigned int y, const float weight)
{
    const unsigned int offset = x + y * m_width;
    SamplePixel *sp = &(m_sampleFrameBuffer->GetPixels()[offset]);
    
    sp->radiance += weight * radiance;
    sp->weight += weight;
}

void Film::UpdateScreenBuffer()
{
    boost::mutex::scoped_lock lock(m_splatMutex);
    
	float scale = 1.0f;
    const SamplePixel *sp = m_sampleFrameBuffer->GetPixels();
    Pixel *p = m_frameBuffer->GetPixels();
    const unsigned int pixelCount = m_width * m_height;
    for (unsigned int i = 0; i < pixelCount; ++i) {
        const float weight = sp[i].weight;
        
        if (weight > 0.f) {
            const float invWeight =  scale / weight;
            
            p[i].r = Radiance2PixelFloat(sp[i].radiance.r * invWeight);
            p[i].g = Radiance2PixelFloat(sp[i].radiance.g * invWeight);
            p[i].b = Radiance2PixelFloat(sp[i].radiance.b * invWeight);
        }
    }

}

float Film::Radiance2PixelFloat(const float x) const {
    
    const float indexf = GammaTableSize * Clamp(x, 0.f, 1.f);
    if (indexf < 0.f)
        return 0.f;
    
    const unsigned int index = Min<unsigned int>((unsigned int) indexf,
                                                 GammaTableSize - 1);
    
    return m_gammaTable[index];
}

void Film::SaveImpl(const std::string &fileName) {
    
    RT_LOG("Saving " << fileName);
    
    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName.c_str());
    if (fif != FIF_UNKNOWN) {
        if ((fif == FIF_HDR) || (fif == FIF_EXR)) {
            FIBITMAP *dib = FreeImage_AllocateT(FIT_RGBF, m_width, m_height, 96);
            
            if (dib) {
                unsigned int pitch = FreeImage_GetPitch(dib);
                BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
                const SampleFrameBuffer *sbe = m_sampleFrameBuffer;
                
                for (unsigned int y = 0; y < m_height; ++y) {
                    FIRGBF *pixel = (FIRGBF *)bits;
                    for (unsigned int x = 0; x < m_width; ++x) {
                        const unsigned int ridx = y * m_width + x;
                        const SamplePixel *sp = sbe->GetPixel(ridx);
                        const float weight = sp->weight;
                        
                        if (weight == 0.f) {
                            pixel[x].red = 0.f;
                            pixel[x].green = 0.f;
                            pixel[x].blue = 0.f;
                        } else {
                            pixel[x].red = sp->radiance.r / weight;
                            pixel[x].green =  sp->radiance.g / weight;
                            pixel[x].blue =  sp->radiance.b / weight;
                        }
                    }
                    
                    // Next line
                    bits += pitch;
                }
                
                if (!FreeImage_Save(fif, dib, fileName.c_str(), 0))
                RT_LOG("Failed image save: " << fileName);
                
                FreeImage_Unload(dib);
            } else
            RT_LOG("Unable to allocate FreeImage HDR image: " << fileName);
        } else {
            FIBITMAP *dib = FreeImage_Allocate(m_width, m_height, 24);
            
            if (dib) {
                unsigned int pitch = FreeImage_GetPitch(dib);
                BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
                const float *pixels = GetScreenBuffer();
                
                for (unsigned int y = 0; y < m_height; ++y) {
                    BYTE *pixel = (BYTE *)bits;
                    for (unsigned int x = 0; x < m_width; ++x) {
                        const int offset = 3 * (x + y * m_width);
                        pixel[FI_RGBA_RED] = (BYTE)(pixels[offset] * 255.f + .5f);
                        pixel[FI_RGBA_GREEN] = (BYTE)(pixels[offset + 1] * 255.f + .5f);
                        pixel[FI_RGBA_BLUE] = (BYTE)(pixels[offset + 2] * 255.f + .5f);
                        pixel += 3;
                    }
                    
                    // Next line
                    bits += pitch;
                }
                
                if (!FreeImage_Save(fif, dib, fileName.c_str(), 0))
                RT_LOG("Failed image save: " << fileName);
                
                FreeImage_Unload(dib);
            } else
            RT_LOG("Unable to allocate FreeImage image: " << fileName);
        }
    } else
    RT_LOG("Image type unknown: " << fileName);
}

