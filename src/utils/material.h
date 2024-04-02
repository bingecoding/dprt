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

#ifndef MATERIAL_H
#define MATERIAL_H

#include "utils/trianglemesh.h"
#include "vector_normal.h"
#include "utils/utils.h"
#include "utils/montecarlo.h"
#include "utils/sampler.h"

enum MaterialType {
    MATTE, AREALIGHT, BLINNPHONG, METAL, MIRROR, ALLOY, MATTEMETAL, PLASTIC
};

inline float AbsCosTheta(const Vector &w) { return fabsf(w.z); }

class Material {
public:
    virtual ~Material() { }
    
    virtual MaterialType GetType() const = 0;
    
    virtual bool IsLightSource() const = 0;
    virtual bool IsDiffuse() const = 0;
    virtual bool IsSpecular() const = 0;
};

class LightMaterial : public Material {
public:
    bool IsLightSource() const { return true; }
    bool IsDiffuse() const { return false; }
    bool IsSpecular() const { return false; }
    
    virtual Spectrum Le(const TriangleMesh *mesh, const unsigned triIndex, const Vector &wo) const = 0;
};

class AreaLightMaterial : public LightMaterial {
public:
    AreaLightMaterial(const Spectrum &col) { m_gain = col; }
    
    MaterialType GetType() const { return AREALIGHT; }
    
    Spectrum Le(const TriangleMesh *mesh, const unsigned triIndex,
                const Vector &wo) const
    {
        // Light sources should be flat
        Normal sampleN = mesh->GetNormal(triIndex, 0);
        
        if (Dot(sampleN, wo) <= 0.f)
            return Spectrum();
        
        return m_gain;
    }
    
    const Spectrum &GetGain() const { return m_gain; }
    
private:
    Spectrum m_gain;
};

class SurfaceMaterial : public Material {
public:
    bool IsLightSource() const { return false; }
    
    virtual Spectrum f(const Vector &wo, const Vector &wi,
                       const Normal &N) const = 0;
    virtual Spectrum Sample_f(const Vector &wo, Vector *wi, const Normal &N,
                              const Normal &shadeN, const float u0,
                              const float u1,  const float u2,
                              const bool onlySpecular, float *pdf,
                              bool &specularBounce) const = 0;

    virtual Spectrum rho(const Vector &wo, const Normal &N,
                         const Normal &shadeN, RandomGenerator *rng) const = 0;
    
    virtual const Spectrum &GetKdOverPI() const = 0;
    
};

class MatteMaterial : public SurfaceMaterial {
public:
    MatteMaterial(const Spectrum &col) {
        m_Kd = col;
        m_KdOverPI = m_Kd * INV_PI;
    }
    
    MaterialType GetType() const { return MATTE; }
    
    bool IsDiffuse() const { return true; }
    bool IsSpecular() const { return false; }
    
    Spectrum f(const Vector &wo, const Vector &wi, const Normal &N) const {
        return m_KdOverPI;
    }

    Spectrum Sample_f(const Vector &wo, Vector *wi, const Normal &N,
                      const Normal &shadeN, const float u0, const float u1,
                      const float u2, const bool onlySpecular,
                      float *pdf, bool &specularBounce) const
    {
        if (onlySpecular) {
            *pdf = 0.f;
            return Spectrum();
        }
        
        Vector dir = CosineSampleHemisphere(u0, u1);
        *pdf = dir.z * INV_PI;
        
        // Local to World
        Vector v1, v2;
        CoordinateSystem(Vector(shadeN), &v1, &v2);
        
        dir = Vector(
                     v1.x * dir.x + v2.x * dir.y + shadeN.x * dir.z,
                     v1.y * dir.x + v2.y * dir.y + shadeN.y * dir.z,
                     v1.z * dir.x + v2.z * dir.y + shadeN.z * dir.z);
        
        (*wi) = dir;
        
        const float dp = Dot(shadeN, *wi);
        // Using 0.0001 instead of 0.0 to cut down fireflies
        if (dp <= 0.0001f) {
            *pdf = 0.f;
            return Spectrum();
        }
        *pdf /=  dp;
        
        specularBounce = false;
        
        return m_KdOverPI;
    }
    
    Spectrum rho(const Vector &wo, const Normal &N,
                 const Normal &shadeN, RandomGenerator *rng) const {
        return m_Kd;
    }
    
    const Spectrum &GetKd() const { return m_Kd; }
    const Spectrum &GetKdOverPI() const { return m_KdOverPI; }
    
private:
    Spectrum m_Kd, m_KdOverPI;
};

class BlinnPhongMaterial : public SurfaceMaterial {
public:
    BlinnPhongMaterial(const Spectrum &col, const Spectrum &spec, const float exponent) :
    m_matte(col), m_Kst(0.f) {
        
        m_Ks = spec;
        m_exponent = 1.f / exponent;
    }
    
    MaterialType GetType() const { return BLINNPHONG; }
    
    bool IsDiffuse() const { return true; }
    bool IsSpecular() const { return true; }
    
    Spectrum f(const Vector &wo, const Vector &wi, const Normal &N) const {
        
        float cosThetaO = AbsCosTheta(wo);
        float cosThetaI = AbsCosTheta(wi);
        if (cosThetaI == 0.f || cosThetaO == 0.f) return Spectrum(0.f);
        Vector wh = wi + wo;
       
        if (wh.x == 0. && wh.y == 0. && wh.z == 0.) return Spectrum(0.f);
        wh = Normalize(wh);
        
        //float costhetah = AbsCosTheta(wh);
        //float D = (m_exponent+2) * INV_TWOPI * powf(costhetah, m_exponent);
        //const Spectrum specular = m_Ks * D;
        
        float specAngle = Max(Dot(wh, Normalize(Vector(N))), 0.f);
        const Spectrum specular = m_Ks * powf(specAngle, m_exponent);
        
        const Spectrum diffuse = m_matte.GetKd() * AbsDot(wi, Normalize(Vector(N)));
        //const Spectrum diffuse = m_matte.GetKdOverPI();
        
        return diffuse + specular;
    }
    
    Spectrum Sample_f(const Vector &wo, Vector *wi, const Normal &N,
                      const Normal &shadeN, const float u0, const float u1,
                      const float u2, const bool onlySpecular,
                      float *pdf, bool &specularBounce) const {
        
        // Compute sampled half-angle vector for Blinn distribution
        float costheta = powf(u0, 1.f / (m_exponent+1));
        float sintheta = sqrtf(Max(0.f, 1.f - costheta*costheta));
        float phi = u1 * 2.f * M_PI;
        Vector wh = SphericalDirection(sintheta, costheta, phi);
        
        // Local to world
        Vector v1, v2;
        CoordinateSystem(Vector(shadeN), &v1, &v2);
        
        wh = Vector(v1.x * wh.x + v2.x * wh.y + shadeN.x * wh.z,
                    v1.y * wh.x + v2.y * wh.y + shadeN.y * wh.z,
                    v1.z * wh.x + v2.z * wh.y + shadeN.z * wh.z);
        
        //if (!SameHemisphere(wo, wh)) wh = -wh;
        
        // Compute incident direction by reflecting about wh
        *wi = -wo + 2.f * Dot(wo, wh) * wh;
        // Compute PDF for from Blinn distribution
        float blinn_pdf = ((m_exponent + 1.f) * powf(costheta, m_exponent)) /
        (2.f * M_PI * 4.f * Dot(wo, wh));
   
        if(Dot(wo, wh) <= 0.f) blinn_pdf = 0.f;
        *pdf = blinn_pdf;
        
        *pdf = 1.f;
        //float costhetah = AbsCosTheta(wh);
        //float D = (m_exponent+2) * INV_TWOPI * powf(costhetah, m_exponent);
        //const Spectrum specular = m_Ks * D;
        
        float specAngle = Max(Dot(wh, Normalize(Vector(N))), 0.f);
        const Spectrum specular = m_Ks * powf(specAngle, m_exponent);
                
        const Spectrum diffuse = m_matte.GetKd() * AbsDot(*wi, Normalize(Vector(N)));
        //const Spectrum diffuse = m_matte.GetKdOverPI();
        //*pdf += SameHemisphere(wo, *wi) ? AbsCosTheta(*wi) * INV_PI : 0.f;
        //*pdf /= 2; // diffuse and specular pdfs, i.e. 2
        
        return diffuse + specular;
    }
    
    // We only return the diffuse part because we only use this for the VPLs
    Spectrum rho(const Vector &wo, const Normal &N, const Normal &shadeN,
                 RandomGenerator *rng) const {
        
        Spectrum r = 0.f;
        int nSamples = 36;
        for(int i=0; i < nSamples; i++) {
            // Estimate one term of $\rho_\roman{hd}$
            Vector wi;
            float pdf = 0.f;
            bool specular = false;
            Spectrum f = this->Sample_f(wo, &wi, N, shadeN, rng->floatValue(),
                                        rng->floatValue(), rng->floatValue(),
                                        false, &pdf, specular);
            
            if (pdf > 0.) r += f * AbsCosTheta(wi) / pdf;
        }
        r /= float(nSamples);
        return r;
    }
    
    Spectrum m_Kst;
    Spectrum m_Ks;
    
    const MatteMaterial &GetMatte() const { return m_matte; }
    const Spectrum &GetKd() const { return m_matte.GetKd(); }
    const Spectrum &GetKdOverPI() const { return m_matte.GetKdOverPI(); }
    const Spectrum GetKs() const { return m_Ks; }
    const float &GetExp() const { return m_exponent; }
    const Spectrum &GetSpecularTerm() const { return m_Kst;}
    void SetSpecularTerm(const Spectrum &spec) { m_Kst = spec;}
    
private:
    MatteMaterial m_matte;
    float m_exponent;
    
};

inline Spectrum FrDiel(float cosi, float cost, const Spectrum &etai,
                const Spectrum &etat) {
    Spectrum Rparl = ((etat * cosi) - (etai * cost)) /
    ((etat * cosi) + (etai * cost));
    Spectrum Rperp = ((etai * cosi) - (etat * cost)) /
    ((etai * cosi) + (etat * cost));
    return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

inline Spectrum FrCond(float cosi, const Spectrum &eta, const Spectrum &k) {
    Spectrum tmp = (eta*eta + k*k) * cosi*cosi;
    Spectrum Rparl2 = (tmp - (2.f * eta * cosi) + 1) /
    (tmp + (2.f * eta * cosi) + 1);
    Spectrum tmp_f = eta*eta + k*k;
    Spectrum Rperp2 =
    (tmp_f - (2.f * eta * cosi) + cosi*cosi) /
    (tmp_f + (2.f * eta * cosi) + cosi*cosi);
    return (Rparl2 + Rperp2) / 2.f;
}

class Fresnel {
public:
    // Fresnel Interface
    virtual ~Fresnel() {}
    virtual Spectrum Evaluate(float cosi) const = 0;
};


class FresnelConductor : public Fresnel {
public:
    // FresnelConductor Public Methods
    Spectrum Evaluate(float cosi) const {
        return FrCond(fabsf(cosi), eta, k);
    }
    FresnelConductor(const Spectrum &e, const Spectrum &kk)
    : eta(e), k(kk) {
    }
    
private:
    Spectrum eta, k;
};


class FresnelDielectric : public Fresnel {
public:
    // FresnelDielectric Public Methods
    Spectrum Evaluate(float cosi) const {
        // Compute Fresnel reflectance for dielectric
        cosi = Clamp(cosi, -1.f, 1.f);
        
        // Compute indices of refraction for dielectric
        bool entering = cosi > 0.;
        float ei = eta_i, et = eta_t;
        if (!entering)
            Swap(ei, et);
        
        // Compute _sint_ using Snell's law
        float sint = ei/et * sqrtf(Max(0.f, 1.f - cosi*cosi));
        if (sint >= 1.) {
            // Handle total internal reflection
            return 1.;
        }
        else {
            float cost = sqrtf(Max(0.f, 1.f - sint*sint));
            Spectrum esi(ei,ei,ei);
            Spectrum est(et,et,et);
            return FrDiel(fabsf(cosi), cost, esi, est);
        }
    }
    FresnelDielectric(float ei, float et) : eta_i(ei), eta_t(et) { }
private:
    float eta_i, eta_t;
};

class FresnelNoOp : public Fresnel {
public:
    Spectrum Evaluate(float) const { return Spectrum(1.); }
};

class MicrofacetDistribution {
public:
    // MicrofacetDistribution Interface
    virtual ~MicrofacetDistribution() { }
    virtual float D(const Vector &wh) const = 0;
    virtual void Sample_f(const Vector &wo, Vector *wi,
                          float u1, float u2, float *pdf) const = 0;
    virtual float Pdf(const Vector &wo, const Vector &wi) const = 0;
};

class Microfacet {
public:
    // Microfacet Public Methods
    Microfacet(const Spectrum &reflectance, Fresnel *f,
               MicrofacetDistribution *d) :
        R(reflectance), distribution(d), fresnel(f)  {
    }
    Spectrum f(const Vector &wo, const Vector &wi) const {
        float cosThetaO = AbsCosTheta(wo);
        float cosThetaI = AbsCosTheta(wi);
        if (cosThetaI == 0.f || cosThetaO == 0.f) return Spectrum(0.f);
        Vector wh = wi + wo;
        if (wh.x == 0. && wh.y == 0. && wh.z == 0.) return Spectrum(0.f);
        wh = Normalize(wh);
        float cosThetaH = Dot(wi, wh);
        Spectrum F = fresnel->Evaluate(cosThetaH);
        return R * distribution->D(wh) * G(wo, wi, wh) * F /
        (4.f * cosThetaI * cosThetaO);
    }

    float G(const Vector &wo, const Vector &wi, const Vector &wh) const {
        float NdotWh = AbsCosTheta(wh);
        float NdotWo = AbsCosTheta(wo);
        float NdotWi = AbsCosTheta(wi);
        float WOdotWh = AbsDot(wo, wh);
        return Min(1.f, Min((2.f * NdotWh * NdotWo / WOdotWh),
                            (2.f * NdotWh * NdotWi / WOdotWh)));
    }
    Spectrum Sample_f(const Vector &wo, Vector *wi,
                                  float u1, float u2, float *pdf) const {
        distribution->Sample_f(wo, wi, u1, u2, pdf);
        if (!SameHemisphere(wo, *wi)) return Spectrum(0.f);
        return f(wo, *wi);
    }
    float Pdf(const Vector &wo, const Vector &wi) const {
        if (!SameHemisphere(wo, wi)) return 0.f;
        return distribution->Pdf(wo, wi);
    }
private:
    // Microfacet Private Data
    Spectrum R;
    MicrofacetDistribution *distribution;
    Fresnel *fresnel;
};


class Blinn : public MicrofacetDistribution {
public:
    Blinn(float e) { if (e > 10000.f || isnan(e)) e = 10000.f;
        exponent = e; }
    // Blinn Public Methods
    float D(const Vector &wh) const {
        float costhetah = AbsCosTheta(wh);
        return (exponent+2) * INV_TWOPI * powf(costhetah, exponent);
    }
    void Sample_f(const Vector &wo, Vector *wi, float u1, float u2,
                         float *pdf) const {
        // Compute sampled half-angle vector $\wh$ for Blinn distribution
        float costheta = powf(u1, 1.f / (exponent+1));
        float sintheta = sqrtf(Max(0.f, 1.f - costheta*costheta));
        float phi = u2 * 2.f * M_PI;
        Vector wh = SphericalDirection(sintheta, costheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
        
        // Compute incident direction by reflecting about $\wh$
        *wi = -wo + 2.f * Dot(wo, wh) * wh;
        
        // Compute PDF for $\wi$ from Blinn distribution
        float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) /
        (2.f * M_PI * 4.f * Dot(wo, wh));
        if (Dot(wo, wh) <= 0.f) blinn_pdf = 0.f;
        *pdf = blinn_pdf;

    }
    float Pdf(const Vector &wo, const Vector &wi) const {
        Vector wh = Normalize(wo + wi);
        float costheta = AbsCosTheta(wh);
        // Compute PDF for $\wi$ from Blinn distribution
        float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) /
        (2.f * M_PI * 4.f * Dot(wo, wh));
        if (Dot(wo, wh) <= 0.f) blinn_pdf = 0.f;
        return blinn_pdf;
    }
private:
    float exponent;
};

class PlasticMaterial : public SurfaceMaterial {
public:
    PlasticMaterial(const Spectrum &col, const Spectrum &spec, const float exponent) :
    m_matte(col) {
        
        m_Ks = spec;
        m_exponent = exponent;
        
        Fresnel *fresnel = new FresnelDielectric(1.5f, 1.f);
        Blinn *blinn = new Blinn(1.f / m_exponent);
        m_blinnMicrofacet = new Microfacet(m_Ks, fresnel, blinn);
        
    }
    
    MaterialType GetType() const { return PLASTIC; }
    
    bool IsDiffuse() const { return true; }
    bool IsSpecular() const { return false; }
    
    Spectrum f(const Vector &wo, const Vector &wi, const Normal &N) const {
        return m_matte.f(wo, wi, N) + m_blinnMicrofacet->f(wo, wi);
    }
    
    Spectrum Sample_f(const Vector &wo, Vector *wi, const Normal &N,
                      const Normal &shadeN, const float u0, const float u1,
                      const float u2, const bool onlySpecular,
                      float *pdf, bool &specularBounce) const
    {
        
        Spectrum f = 0.f;
        unsigned int brdfComponents = 2;
        // Choose sampler randomly
        unsigned int which = Min(Floor2UInt(u2 * brdfComponents), brdfComponents - 1);
        if(which == 0) {
        
            // We use the matte sampler to determine the direction, i.e. wi
            m_matte.Sample_f(wo, wi, N, shadeN, u0, u1, u2, onlySpecular, pdf,
                             specularBounce);
            
            f = this->f(wo, *wi, N);
    
            *pdf += m_blinnMicrofacet->Pdf(wo, *wi);
            *pdf /= (float) brdfComponents;
            
        } else {
            
            m_blinnMicrofacet->Sample_f(wo, wi, u0, u1, pdf);
            
            f = this->f(wo, *wi, N);
            
            *pdf += SameHemisphere(wo, *wi) ? AbsCosTheta(*wi) * INV_PI : 0.f;
            *pdf /= (float) brdfComponents;
        }
        return f;
    }
    
    Spectrum rho(const Vector &wo, const Normal &N, const Normal &shadeN,
                 RandomGenerator *rng) const {
        
        Spectrum r = 0.f;
        int nSamples = 36;
        for(int i=0; i < nSamples; i++) {
            // Estimate one term of $\rho_\roman{hd}$
            Vector wi;
            float pdf = 0.f;
            
            Spectrum f = m_blinnMicrofacet->Sample_f(wo, &wi, rng->floatValue(),
                                                     rng->floatValue(), &pdf);
            if (pdf > 0.) r += f * AbsCosTheta(wi) / pdf;
        }
        r /= float(nSamples);
        return r;
    }
    
    const Spectrum &GetKdOverPI() const { return m_matte.GetKdOverPI(); }

private:
    MatteMaterial m_matte;
    Spectrum m_Ks;
    Microfacet *m_blinnMicrofacet;
    float m_exponent;
};

#endif
