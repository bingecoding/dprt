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

#ifndef RENDERENGINE_H_
#define RENDERENGINE_H_

#include <vector>

#include "scene.h"
#include "engines/gputypes.h"
#include "film.h"

#include "lighttree.h"

enum RenderEngineType {
	PATHCPU, VPLCPU, LIGHTCUTSCPU, RECONSTRUCTIONCUTSCPU,
    PATHGPU, VPLGPU, LIGHTCUTSGPU, RECONSTRUCTIONCUTSGPU, NORENDERENGINE
};

class RenderEngine {
public:
	    
    RenderEngine(Scene *scene, Film *film, boost::mutex *filmMutex,
                 const Properties &cfg) : m_cfg(cfg) {
        m_scene = scene;
        m_film = film;
        m_filmMutex = filmMutex;
        m_started = false;
        
        m_seed = (unsigned int)(WallClockTime() / 1000.0);
        m_rnd = new RandomGenerator(m_seed);
        
        m_lightPaths = m_cfg.GetInt("light.paths");
        m_depth = m_cfg.GetInt("light.depth");
        
    };
    
	virtual ~RenderEngine() { };
    
	virtual void Start() {
		assert (!m_started);
		m_started = true;
	}
    virtual void Interrupt() {};
	virtual void Stop() {
		assert(m_started);
		m_started = false;
	}
    
    virtual unsigned int GetPass() const {return 0; }
    virtual unsigned long long GetSamplesCount() const {return 0; }
    virtual unsigned int GetThreadCount() const { return 0;}
    virtual RenderEngineType GetEngineType()  const { return NORENDERENGINE;}
    
protected:
    
    void Preprocess();
    
	Scene *m_scene;
	Film *m_film;
    const Properties &m_cfg;
	boost::mutex *m_filmMutex;
    
	bool m_started;
    
    LightTree *m_lightTree;
    Sampler *m_sampler;
    RandomGenerator *m_rnd;
    std::vector<VPL> m_virtualLights;
    
    unsigned int m_seed;
    int m_lightPaths;
    int m_depth;

};

Spectrum EstimateDirect(Scene *scene, const Film *film, const Ray &pathRay,
                        const RayHit &rayHit, const SurfaceMaterial *triSurfMat,
                        const Point &hitpoint, const Normal &shadeN,
                        RandomGenerator *rnd, bool *skipVpls);

#endif
