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

#include <string>
#include <fstream>

#include <boost/algorithm/string.hpp>

#include "utils/properties.h"
#include "utils/utils.h"

void Properties::LoadFile(const std::string &fileName)
{
    
    RT_LOG("Loading properties from configuration file: " << fileName);
    
    std::ifstream configFile(fileName);
    if (!configFile)
    {
        throw std::runtime_error("Cannot open config file " + fileName);
    }
    
    // Setup options.
    po::options_description desc("Options");
    SetupOptions(desc);
    
    // Clear map
    po::variables_map vm = po::variables_map();
    try {
        po::store(po::parse_config_file(configFile, desc, true), vm);
    }catch(std::exception err) {
        std::stringstream ss;
        ss << "There is probably a syntax error in the config file";
        throw std::runtime_error(ss.str());
    }
    po::notify(vm);
    
    RT_LOG("Configuration options:");
    
    for (auto iter= vm.begin(); iter != vm.end(); ++iter) {
        std::string key = iter->first;
        std::string value = iter->second.as<std::string>();
        
        m_props[key] = value;
        
        RT_LOG(key << " = " << value);
    }
    
}

std::vector<float> Properties::GetFloatVector(const std::string& paramName) const
{
    
    std::string value = FindParamValue(paramName);
    
    const std::vector<float> vf = ConvertToFloatVector(value);
    
    if(vf.size() != VecSize) {
        std::stringstream ss;
        ss << "Syntax error in " << paramName << " requires " << VecSize <<
        " arguments";
        throw std::runtime_error(ss.str());
    }
    
    return vf;
}

int Properties::GetInt(const std::string &paramName) const
{
    std::string value = FindParamValue(paramName);
    
    return atoi(value.c_str());
}

float Properties::GetFloat(const std::string &paramName) const
{
    std::string value = FindParamValue(paramName);
    
    return atof(value.c_str());
}

std::string Properties::GetString(const std::string &paramName) const
{
    std::string value = FindParamValue(paramName);
    
    return value;
}

std::string Properties::FindParamValue(const std::string &paramName) const
{
    auto it = m_props.find(paramName);
    
    if(it == m_props.end()) {
        std::stringstream ss;
        ss << "Cannot find parameter " << paramName;
        throw std::runtime_error(ss.str());
    }
    
    if(it->second.compare("") == 0) {
        std::stringstream ss;
        ss << "Syntax error in " << paramName << " no arguments";
        throw std::runtime_error(ss.str());
    }
    
    return it->second;
}

void Properties::SetupOptions(po::options_description &desc)
{
    desc.add_options()
    ("camera.lookat.origin", po::value<std::string>())
    ("camera.lookat.target", po::value<std::string>())
    ("camera.lookat.up", po::value<std::string>()->default_value("0.0 0.0 1.0"))
    ("camera.fieldofview", po::value<std::string>()->default_value("45"))
    ("image.width", po::value<std::string>()->default_value("640"))
    ("image.height", po::value<std::string>()->default_value("480"))
    ("scene.file", po::value<std::string>())
    ("scene.materials" , po::value<std::string>())
    ("path.sampler.spp", po::value<std::string>()->default_value("4"))
    ("path.maxdepth", po::value<std::string>()->default_value("3"))
    ("path.russianroulette.depth", po::value<std::string>()->default_value("2"))
    ("path.russianroulette.cap", po::value<std::string>()->default_value("0.125"))
    ("opencl.platform.index", po::value<std::string>()->default_value("0"))
    ("vpl.clamping", po::value<std::string>()->default_value("0.1"))
    ("opencl.devices.select", po::value<std::string>()->default_value("-"))
    ("image.filename", po::value<std::string>()->default_value("image.png"))
    ("batch.halttime", po::value<std::string>()->default_value("0"))
    ("batch.haltspp", po::value<std::string>()->default_value("0"))
    ("light.paths", po::value<std::string>()->default_value("0"))
    ("light.depth", po::value<std::string>()->default_value("0"))
    ("opencl.workgroup.size", po::value<std::string>()->default_value("64"))
    ("opencl.global.work.size", po::value<std::string>()->default_value("1024"))
    ("light.tree.cut.size", po::value<std::string>()->default_value("300"))
    ("renderengine.type", po::value<std::string>()->default_value("1"));
}

std::vector<float> Properties::ConvertToFloatVector(const std::string &values)
{
    
    std::vector<std::string> strs;
    boost::split(strs, values, boost::is_any_of("\t "));

    std::vector<float> vecFloats;
    for (auto it = strs.begin(); it != strs.end(); ++it) {
        if (it->length() != 0)
            vecFloats.push_back(atof(it->c_str()));
        
    }
    
    return vecFloats;
}

