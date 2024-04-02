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

#if defined (WIN32)
#include <windows.h>
#endif

#include <FreeImage.h>

#include "raytracer.h"
#include "spectrum.h"
#include "utils/texmap.h"

TextureMap::TextureMap(const std::string &fileName)
{
	RT_LOG("Reading texture map: " << fileName);

	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(fileName.c_str(), 0);
	if(fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(fileName.c_str());

	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		FIBITMAP *dib = FreeImage_Load(fif, fileName.c_str(), 0);

		if (!dib)
			throw std::runtime_error("Unable to read texture map: " + fileName);

		width = FreeImage_GetWidth(dib);
		height = FreeImage_GetHeight(dib);

		unsigned int pitch = FreeImage_GetPitch(dib);
		FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		BYTE *bits = (BYTE *)FreeImage_GetBits(dib);

		if ((imageType == FIT_RGBAF) && (bpp == 128)) {
			RT_LOG("HDR RGB (128bit) texture map size: " << width << "x" << height << " (" <<
					width * height * sizeof(Spectrum) / 1024 << "Kbytes)");
			pixels = new Spectrum[width * height];
			alpha = NULL;

			for (unsigned int y = 0; y < height; ++y) {
				FIRGBAF *pixel = (FIRGBAF *)bits;
				for (unsigned int x = 0; x < width; ++x) {
					const unsigned int offset = x + (height - y - 1) * width;
					pixels[offset].r = pixel[x].red;
					pixels[offset].g = pixel[x].green;
					pixels[offset].b = pixel[x].blue;
				}

				// Next line
				bits += pitch;
			}
		} else if ((imageType == FIT_RGBF) && (bpp == 96)) {
			RT_LOG("HDR RGB (96bit) texture map size: " << width << "x" << height << " (" <<
					width * height * sizeof(Spectrum) / 1024 << "Kbytes)");
			pixels = new Spectrum[width * height];
			alpha = NULL;

			for (unsigned int y = 0; y < height; ++y) {
				FIRGBF *pixel = (FIRGBF *)bits;
				for (unsigned int x = 0; x < width; ++x) {
					const unsigned int offset = x + (height - y - 1) * width;
					pixels[offset].r = pixel[x].red;
					pixels[offset].g = pixel[x].green;
					pixels[offset].b = pixel[x].blue;
				}

				// Next line
				bits += pitch;
			}
		} else if ((imageType == FIT_BITMAP) && (bpp == 32)) {
			RT_LOG("RGBA texture map size: " << width << "x" << height << " (" <<
					width * height * (sizeof(Spectrum) + sizeof(float)) / 1024 << "Kbytes)");
			const unsigned int pixelCount = width * height;
			pixels = new Spectrum[pixelCount];
			alpha = new float[pixelCount];

			for (unsigned int y = 0; y < height; ++y) {
				BYTE *pixel = (BYTE *)bits;
				for (unsigned int x = 0; x < width; ++x) {
					const unsigned int offset = x + (height - y - 1) * width;
					pixels[offset].r = pixel[FI_RGBA_RED] / 255.f;
					pixels[offset].g = pixel[FI_RGBA_GREEN] / 255.f;
					pixels[offset].b = pixel[FI_RGBA_BLUE] / 255.f;
					alpha[offset] = pixel[FI_RGBA_ALPHA] / 255.f;
					pixel += 4;
				}

				// Next line
				bits += pitch;
			}
		} else if (bpp == 24) {
			RT_LOG("RGB texture map size: " << width << "x" << height << " (" <<
					width * height * sizeof(Spectrum) / 1024 << "Kbytes)");
			pixels = new Spectrum[width * height];
			alpha = NULL;

			for (unsigned int y = 0; y < height; ++y) {
				BYTE *pixel = (BYTE *)bits;
				for (unsigned int x = 0; x < width; ++x) {
					const unsigned int offset = x + (height - y - 1) * width;
					pixels[offset].r = pixel[FI_RGBA_RED] / 255.f;
					pixels[offset].g = pixel[FI_RGBA_GREEN] / 255.f;
					pixels[offset].b = pixel[FI_RGBA_BLUE] / 255.f;
					pixel += 3;
				}

				// Next line
				bits += pitch;
			}
        } 
        else
			throw std::runtime_error("Unsupported bitmap depth in a texture map: " + fileName);

		FreeImage_Unload(dib);
	} else
		throw std::runtime_error("Unknown image file format: " + fileName);

	DuDv.u = 1.f / width;
	DuDv.v = 1.f / height;
}

TextureMap::TextureMap(Spectrum *cols, const unsigned int w, const unsigned int h) {
	pixels = cols;
	alpha = NULL;
	width = w;
	height = h;

	DuDv.u = 1.f / width;
	DuDv.v = 1.f / height;
}

TextureMap::~TextureMap() {
	delete[] pixels;
	delete[] alpha;
}

TextureMapCache::~TextureMapCache() {
	for (size_t i = 0; i < texInstances.size(); ++i)
		delete texInstances[i];
	for (size_t i = 0; i < bumpInstances.size(); ++i)
		delete bumpInstances[i];
	for (size_t i = 0; i < normalInstances.size(); ++i)
		delete normalInstances[i];

	for (std::map<std::string, TextureMap *>::const_iterator it = maps.begin(); it != maps.end(); ++it)
		delete it->second;
}

TextureMap *TextureMapCache::GetTextureMap(const std::string &fileName) {
	// Check if the texture map has already been loaded
	std::map<std::string, TextureMap *>::const_iterator it = maps.find(fileName);

	if (it == maps.end()) {
		// We have yet to load the file
		TextureMap *tm = new TextureMap(fileName);
		maps.insert(std::make_pair(fileName, tm));
		return tm;
	} else {
		return it->second;
	}
}

TexMapInstance *TextureMapCache::GetTexMapInstance(const std::string &fileName) {
	TextureMap *tm = GetTextureMap(fileName);
	TexMapInstance *texm = new TexMapInstance(tm);
	texInstances.push_back(texm);

	return texm;
}

BumpMapInstance *TextureMapCache::GetBumpMapInstance(const std::string &fileName, const float scale) {
	TextureMap *tm = GetTextureMap(fileName);
	BumpMapInstance *bm = new BumpMapInstance(tm, scale);
	bumpInstances.push_back(bm);

	return bm;
}

NormalMapInstance *TextureMapCache::GetNormalMapInstance(const std::string &fileName) {
	TextureMap *tm = GetTextureMap(fileName);
	NormalMapInstance *nm = new NormalMapInstance(tm);
	normalInstances.push_back(nm);

	return nm;
}

void TextureMapCache::GetTexMaps(std::vector<TextureMap *> &tms) {
	for (std::map<std::string, TextureMap *>::const_iterator it = maps.begin(); it != maps.end(); ++it)
		tms.push_back(it->second);
}
