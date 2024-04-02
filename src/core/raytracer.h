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

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <sstream>

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

extern void (*RT_DebugHandler)(const char *msg);

#define RT_LOG(a) { if (RT_DebugHandler) { std::stringstream _RT_LOG_LOCAL_SS; _RT_LOG_LOCAL_SS << a; RT_DebugHandler(_RT_LOG_LOCAL_SS.str().c_str()); } }

#endif
