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

#ifndef BSPHERE_H
#define BSPHERE_H

#include "geometry/point.h"

class BSphere {
public:
	    
	BSphere() : center(0.f, 0.f, 0.f) {
		rad = 0.f;
	}
    
	BSphere(const Point &c, const float r) : center(c) {
		rad = r;
	}
    
	Point center;
	float rad;
};

inline std::ostream &operator<<(std::ostream &os, const BSphere &s) {
	os << "BSphere[" << s.center << ", " << s.rad << "]";
	return os;
}
    
#endif
