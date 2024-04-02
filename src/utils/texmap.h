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

#ifndef TEXMAP_H
#define	TEXMAP_H

#include <string>
#include <vector>
#include <map>

#include "uv.h"
#include "utils.h"

class TextureMap {
public:
	TextureMap(const std::string &fileName);
	TextureMap(Spectrum *cols, const unsigned int w, const unsigned int h);
	~TextureMap();

	const Spectrum GetColor(const UV &uv) const {
		const float s = uv.u * width - 0.5f;
		const float t = uv.v * height - 0.5f;

		const int s0 = Floor2Int(s);
		const int t0 = Floor2Int(t);

		const float ds = s - s0;
		const float dt = t - t0;

		const float ids = 1.f - ds;
		const float idt = 1.f - dt;

		return ids * idt * GetTexel(s0, t0) +
				ids * dt * GetTexel(s0, t0 + 1) +
				ds * idt * GetTexel(s0 + 1, t0) +
				ds * dt * GetTexel(s0 + 1, t0 + 1);
	}

	const bool HasAlpha() const { return alpha != NULL; }
	float GetAlpha(const UV &uv) const {
		assert (alpha != NULL);

		const float s = uv.u * width - 0.5f;
		const float t = uv.v * height - 0.5f;

		const int s0 = Floor2Int(s);
		const int t0 = Floor2Int(t);

		const float ds = s - s0;
		const float dt = t - t0;

		const float ids = 1.f - ds;
		const float idt = 1.f - dt;

		return ids * idt * GetAlphaTexel(s0, t0) +
				ids * dt * GetAlphaTexel(s0, t0 + 1) +
				ds * idt * GetAlphaTexel(s0 + 1, t0) +
				ds * dt * GetAlphaTexel(s0 + 1, t0 + 1);
	}

	const UV &GetDuDv() const {
		return DuDv;
	}

	unsigned int GetWidth() const { return width; }
	unsigned int GetHeight() const { return height; }
	const Spectrum *GetPixels() const { return pixels; };
	const float *GetAlphas() const { return alpha; };

private:
	const Spectrum &GetTexel(const int s, const int t) const {
		const unsigned int u = Mod<int>(s, width);
		const unsigned int v = Mod<int>(t, height);

		const unsigned index = v * width + u;
		assert (index >= 0);
		assert (index < width * height);

		return pixels[index];
	}

	const float &GetAlphaTexel(const unsigned int s, const unsigned int t) const {
		const unsigned int u = Mod(s, width);
		const unsigned int v = Mod(t, height);

		const unsigned index = v * width + u;
		assert (index >= 0);
		assert (index < width * height);

		return alpha[index];
	}

	unsigned int width, height;
	Spectrum *pixels;
	float *alpha;
	UV DuDv;
};

class TexMapInstance {
public:
	TexMapInstance(const TextureMap *tm) : texMap(tm) { }

	const TextureMap *GetTexMap() const { return texMap; }

private:
	const TextureMap *texMap;
};

class BumpMapInstance {
public:
	BumpMapInstance(const TextureMap *tm, const float valueScale) :
		texMap(tm), scale(valueScale) { }

	const TextureMap *GetTexMap() const { return texMap; }
	float GetScale() const { return scale; }

private:
	const TextureMap *texMap;
	const float scale;
};

class NormalMapInstance {
public:
	NormalMapInstance(const TextureMap *tm) : texMap(tm) { }

	const TextureMap *GetTexMap() const { return texMap; }

private:
	const TextureMap *texMap;
};

class TextureMapCache {
public:
    TextureMapCache(){};
	~TextureMapCache();

	TexMapInstance *GetTexMapInstance(const std::string &fileName);
	BumpMapInstance *GetBumpMapInstance(const std::string &fileName, const float scale);
	NormalMapInstance *GetNormalMapInstance(const std::string &fileName);

	void GetTexMaps(std::vector<TextureMap *> &tms);

private:
	TextureMap *GetTextureMap(const std::string &fileName);

	std::map<std::string, TextureMap *> maps;
	std::vector<TexMapInstance *> texInstances;
	std::vector<BumpMapInstance *> bumpInstances;
	std::vector<NormalMapInstance *> normalInstances;
};

#endif
