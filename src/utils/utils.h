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

#ifndef UTILS_H
#define UTILS_H

#include <cmath>

#include <sstream>
#include <fstream>

#include <boost/thread/thread.hpp>

#include "raytracer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef INFINITY
#define INFINITY (std::numeric_limits<float>::infinity())
#endif

#ifndef INV_PI
#define INV_PI  0.31830988618379067154f
#endif

#ifndef INV_TWOPI
#define INV_TWOPI  0.15915494309189533577f
#endif

static const size_t VecSize = 3;

template<class T> inline T Clamp(T val, T low, T high) {
    return val > low ? (val < high ? val : high) : low;
}

template<class T> inline void Swap(T &a, T &b) {
    const T tmp = a;
    a = b;
    b = tmp;
}

template<class T> inline T Max(T a, T b) {
    return a > b ? a : b;
}

template<class T> inline T Min(T a, T b) {
    return a < b ? a : b;
}

inline float Radians(float deg) {
    return (M_PI / 180.f) * deg;
}

inline float Degrees(float rad) {
    return (180.f / M_PI) * rad;
}

template <class T> inline T RoundUp(const T a, const T b) {
    const unsigned int r = a % b;
    if (r == 0)
        return a;
    else
        return a + b - r;
}

inline unsigned int Floor2UInt(float val) {
    return val > 0.f ? static_cast<unsigned int> (floorf(val)) : 0;
}

inline int Floor2Int(double val) {
    return static_cast<int> (floor(val));
}

template<class T> inline T Mod(T a, T b) {
    if (b == 0)
        b = 1;
    
    a %= b;
    if (a < 0)
        a += b;
    
    return a;
}

inline double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__) || defined(__CYGWIN__)
    struct timeval t;
    gettimeofday(&t, NULL);
    
    return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined (WIN32)
    return GetTickCount() / 1000.0;
#else
#error "Unsupported Platform !!!"
#endif
}

inline bool SetThreadRRPriority(boost::thread *thread, int pri = 0) {
#if defined (__linux__) || defined (__APPLE__) || defined(__CYGWIN__)
    {
        const pthread_t tid = (pthread_t)thread->native_handle();
        
        int policy = SCHED_FIFO;
        int sysMinPriority = sched_get_priority_min(policy);
        struct sched_param param;
        param.sched_priority = sysMinPriority + pri;
        
        return pthread_setschedparam(tid, policy, &param);
    }
#elif defined (WIN32)
    {
        const HANDLE tid = (HANDLE)thread->native_handle();
        if (!SetPriorityClass(tid, HIGH_PRIORITY_CLASS))
            return false;
        else
            return true;
        
        /*if (!SetThreadPriority(tid, THREAD_PRIORITY_HIGHEST))
         return false;
         else
         return true;*/
    }
#endif
}

static std::string readFile(const std::string &fileName)
{
	
    std::ifstream file(fileName, std::ifstream::in | std::ifstream::ate);
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
	unsigned size = file.tellg();
	char *data = new char[size];
    
	file.seekg(0, std::ifstream::beg);
	file.read(data, size);
	file.close();
    
	std::string ret(data, size);
	delete[] data;
	return ret;
}


static cl::Program getProgram(const cl::Context &ctx,
                              const std::vector<std::string> &files
                              )
{
    
	std::vector<std::pair<const char *, size_t>> sources;
	std::vector<std::string> data;
    
	//for(const std::string &file : files) {
    for(auto file = files.begin(); file != files.end(); file++) {
        std::string content;
        try{
            content = readFile(*file);
        }catch(std::ifstream::failure err) {
            throw std::runtime_error("Cannot read file " + *file);
        }
		data.push_back(content);
		sources.push_back(std::make_pair(data.back().c_str(),
                                         (size_t)data.back().length()));
	}
    
    cl::Program prog = cl::Program(ctx, sources);
    
    return prog;
}

#endif
