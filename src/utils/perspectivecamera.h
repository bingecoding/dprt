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

#ifndef PERSPECTIVECAMERA_H
#define PERSPECTIVECAMERA_H

#include "transform.h"
#include "utils/montecarlo.h"

class PerspectiveCamera {
public:

    PerspectiveCamera(const Point &o, const Point &t, const Vector &u) :
    m_orig(o), m_target(t), m_up(Normalize(u)), m_fieldOfView(45.f),
    m_clipHither(1e-3f), m_clipYon(1e30f), m_lensRadius(0.f),
    m_focalDistance(10.f)
    {
    }
    
    void Update(const unsigned int filmWidth, const unsigned int filmHeight) {
        
        // right handed coordinate system - default z is up like blender
        m_dir = m_target - m_orig;
        m_dir = Normalize(m_dir);
        
        m_x = Cross(m_dir, m_up);
        m_x = Normalize(m_x);
        
        m_y = Cross(m_x, m_dir);
        m_y = Normalize(m_y);
        
        Transform WorldToCamera = LookAt(m_orig, m_target, m_up);
        m_CameraToWorld = WorldToCamera.GetInverse();
        
        Transform CameraToScreen = Perspective(m_fieldOfView, m_clipHither, m_clipYon);
        
        const float frame =  float(filmWidth) / float(filmHeight);
        float screen[4];
        if (frame < 1.f) {
            screen[0] = -frame;
            screen[1] = frame;
            screen[2] = -1.f;
            screen[3] = 1.f;
        } else {
            screen[0] = -1.f;
            screen[1] = 1.f;
            screen[2] = -1.f / frame;
            screen[3] = 1.f / frame;
        }
        Transform ScreenToRaster =
        Scale(float(filmWidth), float(filmHeight), 1.f) *
        Scale(1.f / (screen[1] - screen[0]), 1.f / (screen[2] - screen[3]), 1.f)
        * Translate(Vector(-screen[0], -screen[3], 0.f));
        
        m_RasterToCamera = CameraToScreen.GetInverse() * ScreenToRaster.GetInverse();
        
    }
    
    void GenerateRay(
                     const float screenX, const float screenY,
                     const unsigned int filmWidth, const unsigned int filmHeight,
                     Ray *ray, const float u1, const float u2, const float u3) const
    {
        
        Transform c2w;
      
        c2w = m_CameraToWorld;
        
        Point Pras(screenX, filmHeight - screenY - 1.f, 0);
        Point Pcamera;
        m_RasterToCamera(Pras, &Pcamera);
        
        ray->o = Pcamera;
        ray->d = Vector(Pcamera.x, Pcamera.y, Pcamera.z);
        
        // Modify ray for depth of field
        if (m_lensRadius > 0.f) {
            // Sample point on lens
            float lensU, lensV;
            ConcentricSampleDisk(u1, u2, &lensU, &lensV);
            lensU *= m_lensRadius;
            lensV *= m_lensRadius;
            
            // Compute point on plane of focus
            const float ft = (m_focalDistance - m_clipHither) / ray->d.z;
            Point Pfocus = (*ray)(ft);
            // Update ray for effect of lens
            ray->o.x += lensU * (m_focalDistance - m_clipHither) / m_focalDistance;
            ray->o.y += lensV * (m_focalDistance - m_clipHither) / m_focalDistance;
            ray->d = Pfocus - ray->o;
        }
        
        ray->d = Normalize(ray->d);
        ray->mint = RAY_EPSILON;
        ray->maxt = (m_clipYon - m_clipHither) / ray->d.z;
        
        c2w(*ray, ray);
    }

    
    const Matrix4x4 GetRasterToCameraMatrix() const {
        return m_RasterToCamera.GetMatrix();
    }
    
    const Matrix4x4 GetCameraToWorldMatrix() const {
        return m_CameraToWorld.GetMatrix();
    }
    
    float GetClipYon() const { return m_clipYon; }
    float GetClipHither() const { return m_clipHither; }
    
    Point m_orig, m_target;
    Vector m_up;
    
    float m_fieldOfView, m_clipHither, m_clipYon, m_lensRadius, m_focalDistance;
    
private:
    Vector m_dir, m_x, m_y;
    Transform m_RasterToCamera, m_CameraToWorld;
    
    Vector m_mbDeltaOrig, m_mbDeltaTarget, m_mbDeltaUp;
};

#endif
