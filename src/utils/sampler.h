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

#ifndef _SAMPLER_H
#define	_SAMPLER_H

#include "utils/randomgen.h"

class Sample;

class Sampler {
public:
    virtual ~Sampler() { }
    
    virtual void Init(const unsigned width, const unsigned height,
                      const unsigned startLine = 0) = 0;
    virtual unsigned int GetPass() = 0;
    
    virtual void GetNextSample(Sample *sample) = 0;
    virtual float GetLazyValue(Sample *sample) = 0;
    
    virtual bool IsLowLatency() const = 0;
    virtual bool IsPreviewOver() const = 0;
};

class Sample {
public:
    Sample() { }
    
    void Init(Sampler *s,
              const float x, const float y,
              const unsigned p) {
        sampler = s;
        
        screenX = x;
        screenY = y;
        pass = p;
    }
    
    float GetLazyValue() {
        return sampler->GetLazyValue(this);
    }
    
    float screenX, screenY;
    unsigned int pass;
    
    private:
    Sampler *sampler;
};

class RandomSampler : public Sampler {
public:
    RandomSampler(const bool lowLat, unsigned long startSeed,
                  const unsigned width, const unsigned height, const unsigned int spp,
                  const unsigned startLine = 0) :
    seed(startSeed), samplePerPixel(spp), samplePerPixel2(spp * spp),
    screenStartLine(startLine), lowLatency(lowLat) {
        rndGen = new RandomGenerator(seed);
        
        Init(width, height, screenStartLine);
    }
    ~RandomSampler() {
        delete rndGen;
    }
    
    void Init(const unsigned width, const unsigned height, const unsigned startLine = 0) {
        screenWidth = width;
        screenHeight = height;
        if (startLine > 0)
        screenStartLine = startLine;
        currentSampleScreenX = 0;
        currentSampleScreenY = screenStartLine;
        currentSubSampleIndex = 0;
        pass = 0;
        
        previewOver = !lowLatency;
        startTime = WallClockTime();
    }
    
    void GetNextSample(Sample *sample) {
        if (!lowLatency || (pass >= 64)) {
            previewOver = true;
            // In order to improve ray coherency
            if (samplePerPixel == 1)
            GetNextSample1x1(sample);
            else
            GetNextSampleNxN(sample);
        } else if (previewOver || (pass >= 32)) {
            previewOver = true;
            GetNextSample1x1(sample);
        } else {
            // In order to update the screen faster for the first 16 passes
            GetNextSamplePreview(sample);
            CheckPreviewOver();
        }
    }
    
    float GetLazyValue(Sample *sample) {
        return rndGen->floatValue();
    }
    
    unsigned int GetPass() { return pass; }
    bool IsLowLatency() const { return lowLatency; }
    bool IsPreviewOver() const { return previewOver; }
    
    private:
    void CheckPreviewOver() {
        if (WallClockTime() - startTime > 2.0)
        previewOver = true;
    }
    
    void GetNextSampleNxN(Sample *sample) {
        const unsigned int stepX = currentSubSampleIndex % samplePerPixel;
        const unsigned int stepY = currentSubSampleIndex / samplePerPixel;
        
        unsigned int scrX = currentSampleScreenX;
        unsigned int scrY = currentSampleScreenY;
        
        currentSubSampleIndex++;
        if (currentSubSampleIndex == samplePerPixel2) {
            currentSubSampleIndex = 0;
            currentSampleScreenX++;
            if (currentSampleScreenX  >= screenWidth) {
                currentSampleScreenX = 0;
                currentSampleScreenY++;
                
                if (currentSampleScreenY >= screenHeight) {
                    currentSampleScreenY = 0;
                    pass += samplePerPixel2;
                }
            }
        }
        
        const float r1 = (stepX + rndGen->floatValue()) / samplePerPixel - .5f;
        const float r2 = (stepY + rndGen->floatValue()) / samplePerPixel - .5f;
        
        sample->Init(this,
                     scrX + r1, scrY + r2,
                     pass);
    }
    
    void GetNextSample1x1(Sample *sample) {
        unsigned int scrX = currentSampleScreenX;
        unsigned int scrY = currentSampleScreenY;
        
        currentSampleScreenX++;
        if (currentSampleScreenX >= screenWidth) {
            currentSampleScreenX = 0;
            currentSampleScreenY++;
            
            if (currentSampleScreenY >= screenHeight) {
                currentSampleScreenY = 0;
                pass++;
            }
        }
        
        const float r1 = rndGen->floatValue() - .5f;
        const float r2 = rndGen->floatValue() - .5f;
        
        sample->Init(this,
                     scrX + r1, scrY + r2,
                     pass);
    }
    
    void GetNextSamplePreview(Sample *sample) {
        unsigned int scrX, scrY;
        for (;;) {
            unsigned int stepX = pass % 4;
            unsigned int stepY = (pass / 4) % 4;
            
            scrX = currentSampleScreenX * 4 + stepX;
            scrY = currentSampleScreenY * 4 + stepY;
            
            currentSampleScreenX++;
            if (currentSampleScreenX * 4 >= screenWidth) {
                currentSampleScreenX = 0;
                currentSampleScreenY++;
                
                if (currentSampleScreenY * 4 >= screenHeight) {
                    currentSampleScreenY = 0;
                    pass++;
                }
            }
            
            // Check if we are inside the screen
            if ((scrX < screenWidth) && (scrY < screenHeight)) {
                // Ok, it is a valid sample
                break;
            } else if (pass >= 32) {
                GetNextSample(sample);
                return;
            }
        }
        
        const float r1 = rndGen->floatValue() - .5f;
        const float r2 = rndGen->floatValue() - .5f;
        
        sample->Init(this,
                     scrX + r1, scrY + r2,
                     pass);
    }
    
    RandomGenerator *rndGen;
    const unsigned long seed;
    const unsigned int samplePerPixel, samplePerPixel2;
    unsigned int screenWidth, screenHeight, screenStartLine;
    
    unsigned int currentSampleScreenX, currentSampleScreenY, currentSubSampleIndex;
    unsigned int pass;
    double startTime;
    bool lowLatency, previewOver;
};

#endif	/* _SAMPLER_H */
