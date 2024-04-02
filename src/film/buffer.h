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

#ifndef BUFFER_H
#define BUFFER_H

#include "film/spectrum.h"

typedef struct {
    float screenX, screenY;
    Spectrum radiance;
} SampleBufferElem;

class SampleBuffer {
public:
    SampleBuffer(const size_t bufferSize) : m_size(bufferSize)
    {
        m_samples = new SampleBufferElem[m_size];
        Reset();
    }
    ~SampleBuffer() { delete [] m_samples; }
    
    void Reset() { m_currentFreeSample = 0; }
    bool IsFull() const { return (m_currentFreeSample >= m_size);}
    
    void SplatSample(const float scrX, const float scrY,
                     const Spectrum &radiance)
    {
        // Safety check
        if (!radiance.IsNaN()) {
            SampleBufferElem *s = &m_samples[m_currentFreeSample++];
            
            s->screenX = scrX;
            s->screenY = scrY;
            s->radiance = radiance;
        }
    }
    
    SampleBufferElem *GetSampleBuffer() const { return m_samples; }
    
    size_t GetSampleCount() const { return m_currentFreeSample; }
    size_t GetSize() const { return m_size; }

    
private:
    size_t m_size;
    size_t m_currentFreeSample;
    
    SampleBufferElem *m_samples;
};

typedef struct {
    Spectrum radiance;
    float weight;
} SamplePixel;

class SampleFrameBuffer {
public:
    SampleFrameBuffer(const unsigned int width, const unsigned int height)
    : m_width(width), m_height(height)
    {
        m_pixels = new SamplePixel[m_width * m_height];
        Clear();
    }
    ~SampleFrameBuffer() {
        delete [] m_pixels;
    }
    
    SamplePixel *GetPixels() const { return m_pixels; }
    
    SamplePixel *GetPixel(const unsigned int index) const {
        assert (index >= 0);
        assert (index < m_width * m_height);
        
        return &m_pixels[index];
    }
    
    void Clear() {
        for (unsigned int i = 0; i < m_width * m_height; ++i) {
            m_pixels[i].radiance.r = 0.f;
            m_pixels[i].radiance.g = 0.f;
            m_pixels[i].radiance.b = 0.f;
            m_pixels[i].weight = 0.f;
        }
    };
    
private:
    const unsigned int m_width, m_height;
    
    SamplePixel *m_pixels;
};

typedef Spectrum Pixel;

class FrameBuffer {
public:
    FrameBuffer(const unsigned int w, const unsigned int h)
    : m_width(w), m_height(h) {
        m_pixels = new Pixel[m_width * m_height];
        
        Clear();
    }
    ~FrameBuffer() {
        delete[] m_pixels;
    }
    
    void Clear() {
        for (unsigned int i = 0; i < m_width * m_height; ++i) {
            m_pixels[i].r = 0.f;
            m_pixels[i].g = 0.f;
            m_pixels[i].b = 0.f;
        }
    };
    
    Pixel *GetPixels() const { return m_pixels; }

private:
    const unsigned int m_width, m_height;
    
    Pixel *m_pixels;
};

#endif
