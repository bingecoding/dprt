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

#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <string>
#include <map>

#include <boost/program_options.hpp>

#include "raytracer.h"

namespace po = boost::program_options;

class Properties {
public:
    
    Properties() { }
    void LoadFile(const std::string &fileName);
    
    std::string FindParamValue(const std::string& paramName) const;
    std::vector<float> GetFloatVector(const std::string& paramName) const;
    int GetInt(const std::string& paramName) const;
    float GetFloat(const std::string& paramName) const;
    std::string GetString(const std::string &paramName) const;
    
    static std::vector<float> ConvertToFloatVector(const std::string &values);
    
private:
    
    void SetupOptions(po::options_description& desc);
    
    std::map<std::string, std::string> m_props;
    
};

#endif
