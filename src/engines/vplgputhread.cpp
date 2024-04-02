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

#include "engines/pathgpu.h"
#include "engines/vplgpu.h"


VPLGPURenderThread::VPLGPURenderThread(const unsigned int index,
                                       const unsigned int seedBase,
                                       const float samplingStart,
                                       OpenCLIntersectionDevice *device,
                                       VPLGPURenderEngine *renderEngine)
{
    m_intersectionDevice = device;
    m_samplingStart = samplingStart;
    m_seed = seedBase;
    m_reportedPermissionError = false;
    
	m_renderThread = NULL;
    
    m_lightPaths = renderEngine->m_cfg.GetInt("light.paths");
    m_depth = renderEngine->m_cfg.GetInt("light.depth");
    
	m_threadIndex = index;
	m_renderEngine = renderEngine;
    
	m_started = false;
	m_frameBuffer = NULL;
    
	m_kernelsParameters = "";
	m_initKernel = NULL;
    

	m_initFBKernel = NULL;
	m_updatePBKernel = NULL;
	m_updatePBBluredKernel = NULL;
	m_advancePathStep1Kernel = NULL;
	m_advancePathStep2Kernel = NULL;
    m_initPreprocessingKernel = NULL;
    m_preprocessingKernel = NULL;
    m_integratorKernel = NULL;
}

VPLGPURenderThread::~VPLGPURenderThread()
{
    if (m_started)
		Stop();
    
	delete m_initKernel;
	delete m_initFBKernel;
	delete m_updatePBKernel;
	delete m_updatePBBluredKernel;
	//delete m_advancePathStep1Kernel;
	//delete m_advancePathStep2Kernel;
    
    delete m_initPreprocessingKernel;
    delete m_preprocessingKernel;
    delete m_integratorKernel;
	
    delete[] m_frameBuffer;
}

static void AppendMatrixDefinition(stringstream &ss, const char *paramName, const Matrix4x4 &m) {
	for (unsigned int i = 0; i < 4; ++i) {
		for (unsigned int j = 0; j < 4; ++j)
			ss << " -D " << paramName << "_" << i << j << "=" << m.m[i][j] << "f";
	}
}

void VPLGPURenderThread::Start()
{
	m_started = true;
    
    m_renderEngine->Preprocess();
	InitRender();
    
    // Create the thread for the rendering
    m_renderThread = new boost::thread(boost::bind(VPLGPURenderThread::RenderThreadImpl, this));
    
    // Set renderThread priority
    bool res = SetThreadRRPriority(m_renderThread);
    if (res && !m_reportedPermissionError) {
        cerr << "[VPLGPURenderThread::" << m_threadIndex << "] Failed to set ray intersection thread priority (you probably need root/administrator permission to set thread realtime priority)" << endl;
        m_reportedPermissionError = true;
    }
    
	m_lastCameraUpdate = WallClockTime();
}

void VPLGPURenderThread::InitRender()
{
    const unsigned int pixelCount = m_renderEngine->m_film->GetWidth() *
    m_renderEngine->m_film->GetHeight();
    
    const int globalWorkGroupSize = m_renderEngine->m_globalWorkGroupSize;
    
    delete[] m_frameBuffer;
    
    m_frameBuffer = new PixelGPU[pixelCount];

    for (unsigned int i = 0; i < pixelCount; ++i) {
		m_frameBuffer[i].c.r = 0.f;
		m_frameBuffer[i].c.g = 0.f;
		m_frameBuffer[i].c.b = 0.f;
		m_frameBuffer[i].count = 0;
	}
    
	Scene *scene = m_renderEngine->m_scene;
    
    const unsigned int startLine = Clamp<unsigned int>(
                                                       m_renderEngine->
                                                       m_film->GetHeight() * m_samplingStart,
                                                       0, m_renderEngine->
                                                       m_film->GetHeight() - 1);
    
    cl::Context &oclContext = m_intersectionDevice->GetOpenCLContext();
	cl::Device &oclDevice = m_intersectionDevice->GetOpenCLDevice();
    
    double tStart, tEnd;
    
    //--------------------------------------------------------------------------
	// Allocate light sample buffers
	//--------------------------------------------------------------------------
    
    const OpenCLIntersectionDevice *deviceDesc = m_intersectionDevice;
    
	tStart = WallClockTime();
  
    //--------------------------------------------------------------------------
	// Allocate vpl buffer
	//--------------------------------------------------------------------------
    
    std::vector<VPL> &vpls = m_renderEngine->m_virtualLights;
    size_t nLights = vpls.size();
    
    const size_t vplGPUSize = m_triLightsBuff ?
    sizeof(VPL) : sizeof(Path);
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] VPL buffer size: " << (vplGPUSize * nLights / 1024) <<
    "Kbytes" << endl;
  
    VPL *vplsFlat = new VPL[nLights];
    for(int i=0; i < vpls.size(); i++) {
        vplsFlat[i].n = vpls[i].n;
        vplsFlat[i].hitPoint = vpls[i].hitPoint;
        vplsFlat[i].contrib = vpls[i].contrib;
    }
    
    m_vplsBuff = new cl::Buffer(oclContext,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(VPL) * nLights,
                                vplsFlat
                                );
    deviceDesc->AllocMemory(m_vplsBuff->getInfo<CL_MEM_SIZE>());
    
    

    //--------------------------------------------------------------------------
	// Allocate frame, ray and hit buffer
	//--------------------------------------------------------------------------
    
    cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] FrameBuffer buffer size: " <<

    (sizeof(PixelGPU) * m_renderEngine->m_film->GetWidth() *
     m_renderEngine->m_film->GetHeight() / 1024) << "Kbytes" << endl;
	m_frameBufferBuff = new cl::Buffer(oclContext,
                                       CL_MEM_READ_WRITE,
                                       sizeof(PixelGPU) *
                                       m_renderEngine->m_film->GetWidth() *
                                       m_renderEngine->m_film->GetHeight());
	deviceDesc->AllocMemory(m_frameBufferBuff->getInfo<CL_MEM_SIZE>());
    
    cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Ray buffer size: " << (sizeof(Ray) * globalWorkGroupSize / 1024) <<
    "Kbytes" << endl;
    
    m_raysBuff = new cl::Buffer(oclContext,
                                CL_MEM_READ_WRITE,
                                sizeof(Ray) * globalWorkGroupSize);
    deviceDesc->AllocMemory(m_raysBuff->getInfo<CL_MEM_SIZE>());
    
    cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] RayHit buffer size: " << (sizeof(RayHit) * globalWorkGroupSize / 1024) <<
    "Kbytes" << endl;
	m_hitsBuff = new cl::Buffer(oclContext,
                                CL_MEM_READ_WRITE,
                                sizeof(RayHit) * globalWorkGroupSize);
	deviceDesc->AllocMemory(m_hitsBuff->getInfo<CL_MEM_SIZE>());

    //--------------------------------------------------------------------------
	// Allocate Mesh ID buffer
	//--------------------------------------------------------------------------
    
    const unsigned int trianglesCount = scene->GetTotalTriangleCount();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] MeshIDs buffer size: " << (sizeof(unsigned int) *
                                  trianglesCount / 1024) << "Kbytes" << endl;
	m_meshIDBuff = new cl::Buffer(oclContext,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(unsigned int) * trianglesCount,
                                  (void *)scene->GetMeshIDTable());
	deviceDesc->AllocMemory(m_meshIDBuff->getInfo<CL_MEM_SIZE>());
    
	tEnd = WallClockTime();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] OpenCL buffer creation time: " << int((tEnd - tStart) * 1000.0) <<
    "ms" << endl;
    
    //--------------------------------------------------------------------------
	// Translate material definitions
	//--------------------------------------------------------------------------
    
	tStart = WallClockTime();
    
    bool enable_MAT_MATTE = false;
	bool enable_MAT_AREALIGHT = false;
    
    const unsigned int materialsCount = scene->m_objectMaterials.size();
	MaterialGPU *mats = new MaterialGPU[materialsCount];
	for (unsigned int i = 0; i < materialsCount; ++i) {
        
        Material *m = scene->m_objectMaterials[i];
		MaterialGPU *gpum = &mats[i];
        
		switch (m->GetType()) {
			case MATTE: {
				enable_MAT_MATTE = true;
				MatteMaterial *mm = (MatteMaterial *)m;
                
				gpum->type = MAT_MATTE;
				gpum->param.matte.r = mm->GetKd().r;
				gpum->param.matte.g = mm->GetKd().g;
				gpum->param.matte.b = mm->GetKd().b;
				break;
			}
            case BLINNPHONG: {
                
                BlinnPhongMaterial *bpm = (BlinnPhongMaterial *)m;
                
                gpum->type = MAT_BLINNPHONG;
                gpum->param.blinnPhong.matte.r = bpm->GetKd().r;
                gpum->param.blinnPhong.matte.g = bpm->GetKd().g;
                gpum->param.blinnPhong.matte.b = bpm->GetKd().b;
                
                gpum->param.blinnPhong.spec_r = bpm->GetKs().r;
                gpum->param.blinnPhong.spec_g = bpm->GetKs().g;
                gpum->param.blinnPhong.spec_b = bpm->GetKs().b;
                
                gpum->param.blinnPhong.exponent = bpm->GetExp();
                
                break;
            }
			case AREALIGHT: {
				enable_MAT_AREALIGHT = true;
				AreaLightMaterial *alm = (AreaLightMaterial *)m;
                
				gpum->type = MAT_AREALIGHT;
				gpum->param.areaLight.gain_r = alm->GetGain().r;
				gpum->param.areaLight.gain_g = alm->GetGain().g;
				gpum->param.areaLight.gain_b = alm->GetGain().b;
				break;
			}
            default: {
				enable_MAT_MATTE = true;
				gpum->type = MAT_MATTE;
				gpum->param.matte.r = 0.75f;
				gpum->param.matte.g = 0.75f;
				gpum->param.matte.b = 0.75f;
				break;
			}
        }
    }
    
    cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Materials buffer size: " <<

    (sizeof(MaterialGPU) * materialsCount / 1024) << "Kbytes" << endl;
	m_materialsBuff = new cl::Buffer(oclContext,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(MaterialGPU) * materialsCount,
                                     mats);
	deviceDesc->AllocMemory(m_materialsBuff->getInfo<CL_MEM_SIZE>());
    
	delete[] mats;
    
    tEnd = WallClockTime();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Material translation time: " << int((tEnd - tStart) * 1000.0) <<
    "ms" << endl;
    
    //--------------------------------------------------------------------------
	// Translate area lights
	//--------------------------------------------------------------------------

    tStart = WallClockTime();
    
	// Count the area lights
	unsigned int areaLightCount = 0;
	for (unsigned int i = 0; i < scene->m_lights.size(); ++i) {
		if (scene->m_lights[i]->IsAreaLight())
			++areaLightCount;
	}
    
    if (areaLightCount > 0) {

		TriangleLightGPU *tals = new TriangleLightGPU[areaLightCount];

		unsigned int index = 0;
		for (unsigned int i = 0; i < scene->m_lights.size(); ++i) {
			if (scene->m_lights[i]->IsAreaLight()) {
				const TriangleLight *tl = (TriangleLight *)scene->m_lights[i];
                
				const TriangleMesh *mesh = scene->m_objectMeshes[tl->GetMeshIndex()];
				const Triangle *tri = &(mesh->GetTriangles()[tl->GetTriIndex()]);
				tals[index].v0 = mesh->GetVertex(tri->v[0]);
				tals[index].v1 = mesh->GetVertex(tri->v[1]);
				tals[index].v2 = mesh->GetVertex(tri->v[2]);
                
				tals[index].normal = mesh->GetNormal(tri->v[0]);
                
				tals[index].area = tl->GetArea();
                
				AreaLightMaterial *alm = (AreaLightMaterial *)tl->GetMaterial();
				tals[index].gain_r = alm->GetGain().r;
				tals[index].gain_g = alm->GetGain().g;
				tals[index].gain_b = alm->GetGain().b;
                
				++index;
			}
		}
        
		cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Triangle lights buffer size: " <<

        (sizeof(TriangleLightGPU) * areaLightCount / 1024) <<
        "Kbytes" << endl;
		m_triLightsBuff = new cl::Buffer(oclContext,
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(TriangleLightGPU) *
                                         areaLightCount,
                                         tals);
		deviceDesc->AllocMemory(m_triLightsBuff->getInfo<CL_MEM_SIZE>());
        
		delete[] tals;
	} else {
		m_triLightsBuff = NULL;
    }
    
	tEnd = WallClockTime();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Area lights translation time: " << int((tEnd - tStart) * 1000.0) <<
    "ms" << endl;

    //--------------------------------------------------------------------------
	// Translate mesh material indices
	//--------------------------------------------------------------------------
    
	tStart = WallClockTime();
    
    const unsigned int meshCount = scene->m_objectMaterials.size();
	unsigned int *meshMats = new unsigned int[meshCount];
	for (unsigned int i = 0; i < meshCount; ++i) {
		Material *m = scene->m_objectMaterials[i];
        
		// Look for the index
		unsigned int index = 0;
		for (unsigned int j = 0; j < materialsCount; ++j) {
			if (m == scene->m_objectMaterials[j]) {
				index = j;
				break;
			}
		}
        
		meshMats[i] = index;
	}
    
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Mesh material index buffer size: " <<
    (sizeof(unsigned int) * meshCount / 1024) << "Kbytes" << endl;
	m_meshMatsBuff = new cl::Buffer(oclContext,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(unsigned int) * meshCount,
                                    meshMats);
	deviceDesc->AllocMemory(m_meshMatsBuff->getInfo<CL_MEM_SIZE>());
    
	delete[] meshMats;
    
	tEnd = WallClockTime();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Material indices translation time: " << int((tEnd - tStart) * 1000.0) <<
    "ms" << endl;
    
    //--------------------------------------------------------------------------
	// Translate mesh colors
	//--------------------------------------------------------------------------
    
	tStart = WallClockTime();
    
	const unsigned int colorsCount = scene->GetTotalVertexCount();
	Spectrum *colors = new Spectrum[colorsCount];
	unsigned int cIndex = 0;
	for (unsigned int i = 0; i < scene->m_objectMeshes.size(); ++i) {
		TriangleMesh *mesh = scene->m_objectMeshes[i];
        
		for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j) {
            colors[cIndex++] = Spectrum(1.f, 1.f, 1.f);
		}
	}
    
    
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Colors buffer size: " << (sizeof(Spectrum) * colorsCount / 1024) <<
    "Kbytes" << endl;
	m_colorsBuff = new cl::Buffer(oclContext,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(Spectrum) * colorsCount,
                                  colors);
	deviceDesc->AllocMemory(m_colorsBuff->getInfo<CL_MEM_SIZE>());
	delete[] colors;
    
    //--------------------------------------------------------------------------
	// Translate mesh normals
	//--------------------------------------------------------------------------
    
	const unsigned int normalsCount = scene->GetTotalVertexCount();
	Normal *normals = new Normal[normalsCount];
	unsigned int nIndex = 0;
	for (unsigned int i = 0; i < scene->m_objectMeshes.size(); ++i) {
		TriangleMesh *mesh = scene->m_objectMeshes[i];
        
		for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
			normals[nIndex++] = mesh->GetNormal(j);
	}
    
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Normals buffer size: " << (sizeof(Normal) * normalsCount / 1024) <<
    "Kbytes" << endl;
	m_normalsBuff = new cl::Buffer(oclContext,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(Normal) * normalsCount,
                                   normals);
	deviceDesc->AllocMemory(m_normalsBuff->getInfo<CL_MEM_SIZE>());
	delete[] normals;
    
	//--------------------------------------------------------------------------
	// Translate mesh indices
	//--------------------------------------------------------------------------
    
	const TriangleMesh *preprocessedMesh;
	switch (scene->GetAcceleratorType()) {
		case ACCEL_BVH:
			preprocessedMesh = ((BVHAccel *)scene->
                                GetAccelerator())->GetPreprocessedMesh();
			break;
		default:
			throw runtime_error("ACCEL_MQBVH is not yet supported by\
                                VPLGPURenderEngine");
	}
    
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Triangles buffer size: " << (sizeof(Triangle) * trianglesCount / 1024) <<
    "Kbytes" << endl;
	m_trianglesBuff = new cl::Buffer(oclContext,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(Triangle) * trianglesCount,
                                     (void *)preprocessedMesh->GetTriangles());
	deviceDesc->AllocMemory(m_trianglesBuff->getInfo<CL_MEM_SIZE>());
    
	tEnd = WallClockTime();
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Mesh information translation time: " << int((tEnd - tStart) * 1000.0) <<
    "ms" << endl;
    
    //--------------------------------------------------------------------------
	// Allocate path buffer
	//--------------------------------------------------------------------------
    
    const size_t pathGPUSize = m_triLightsBuff ?
    sizeof(PathDL) : sizeof(Path);
	cerr << "[VPLGPURenderThread::" << m_threadIndex <<
    "] Paths buffer size: " << (pathGPUSize * globalWorkGroupSize / 1024) <<
    "Kbytes" << endl;
    m_pathsBuff = new cl::Buffer(oclContext,
                                 CL_MEM_READ_WRITE,
                                 pathGPUSize * globalWorkGroupSize);
    deviceDesc->AllocMemory(m_pathsBuff->getInfo<CL_MEM_SIZE>());

    //--------------------------------------------------------------------------
    // Translate mesh texture maps
    //--------------------------------------------------------------------------
    
    std::vector<TextureMap *> tms;
    scene->m_texMapCache->GetTexMaps(tms);
    // Calculate the amount of ram to allocate
    unsigned int totRGBTexMem = 0;
    unsigned int totAlphaTexMem = 0;
    
    for (unsigned int i = 0; i < tms.size(); ++i) {
        TextureMap *tm = tms[i];
        const unsigned int pixelCount = tm->GetWidth() * tm->GetHeight();
        
        totRGBTexMem += pixelCount;
        if (tm->HasAlpha())
        totAlphaTexMem += pixelCount;
    }
    
    if ((totRGBTexMem > 0) || (totAlphaTexMem > 0)) {
        TexMap *gpuTexMap = new TexMap[tms.size()];
        
        if (totRGBTexMem > 0) {
            unsigned int rgbOffset = 0;
            Spectrum *rgbTexMem = new Spectrum[totRGBTexMem];
            
            for (unsigned int i = 0; i < tms.size(); ++i) {
                TextureMap *tm = tms[i];
                const unsigned int pixelCount = tm->GetWidth() * tm->GetHeight();
                
                memcpy(&rgbTexMem[rgbOffset], tm->GetPixels(), pixelCount * sizeof(Spectrum));
                gpuTexMap[i].rgbOffset = rgbOffset;
                rgbOffset += pixelCount;
            }
            
            cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] TexMap buffer size: " << (sizeof(Spectrum) * totRGBTexMem / 1024) << "Kbytes" << endl;
            m_texMapRGBBuff = new cl::Buffer(oclContext,
                                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             sizeof(Spectrum) * totRGBTexMem,
                                             rgbTexMem);
            deviceDesc->AllocMemory(m_texMapRGBBuff->getInfo<CL_MEM_SIZE>());
            
            delete[] rgbTexMem;
        } else
        m_texMapRGBBuff = NULL;
        
        if (totAlphaTexMem > 0) {
            unsigned int alphaOffset = 0;
            float *alphaTexMem = new float[totAlphaTexMem];
            
            for (unsigned int i = 0; i < tms.size(); ++i) {
                TextureMap *tm = tms[i];
                const unsigned int pixelCount = tm->GetWidth() * tm->GetHeight();
                
                if (tm->HasAlpha()) {
                    memcpy(&alphaTexMem[alphaOffset], tm->GetAlphas(), pixelCount * sizeof(float));
                    gpuTexMap[i].alphaOffset = alphaOffset;
                    alphaOffset += pixelCount;
                } else
                gpuTexMap[i].alphaOffset = 0xffffffffu;
            }
            
            cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] TexMap buffer size: " << (sizeof(float) * totAlphaTexMem / 1024) << "Kbytes" << endl;
            m_texMapAlphaBuff = new cl::Buffer(oclContext,
                                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               sizeof(float) * totAlphaTexMem,
                                               alphaTexMem);
            deviceDesc->AllocMemory(m_texMapAlphaBuff->getInfo<CL_MEM_SIZE>());
            
            delete[] alphaTexMem;
        } else
        m_texMapAlphaBuff = NULL;
        
        // Translate texture map description
        for (unsigned int i = 0; i < tms.size(); ++i) {
            TextureMap *tm = tms[i];
            gpuTexMap[i].width = tm->GetWidth();
            gpuTexMap[i].height = tm->GetHeight();
        }
        
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] TexMap indices buffer size: " << (sizeof(TexMap) * tms.size() / 1024) << "Kbytes" << endl;
        m_texMapDescBuff = new cl::Buffer(oclContext,
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(TexMap) * tms.size(),
                                          gpuTexMap);
        deviceDesc->AllocMemory(m_texMapDescBuff->getInfo<CL_MEM_SIZE>());
        delete[] gpuTexMap;
        
        // Translate mesh texture indices
        unsigned int *meshTexs = new unsigned int[meshCount];
        for (unsigned int i = 0; i < meshCount; ++i) {
            TexMapInstance *t = scene->m_objectTexMaps[i];
            
            if (t) {
                // Look for the index
                unsigned int index = 0;
                for (unsigned int j = 0; j < tms.size(); ++j) {
                    if (t->GetTexMap() == tms[j]) {
                        index = j;
                        break;
                    }
                }
                
                meshTexs[i] = index;
            } else
            meshTexs[i] = 0xffffffffu;
        }
        
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Mesh texture maps index buffer size: " << (sizeof(unsigned int) * meshCount / 1024) << "Kbytes" << endl;
        m_meshTexsBuff = new cl::Buffer(oclContext,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(unsigned int) * meshCount,
                                        meshTexs);
        deviceDesc->AllocMemory(m_meshTexsBuff->getInfo<CL_MEM_SIZE>());
        
        delete[] meshTexs;
        
        // Translate vertex uvs
        const unsigned int uvsCount = scene->GetTotalVertexCount();
        UV *uvs = new UV[uvsCount];
        unsigned int uvIndex = 0;
        for (unsigned int i = 0; i < scene->m_objectMeshes.size(); ++i) {
            TriangleMesh *mesh = scene->m_objectMeshes[i];
            
            for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j) {
                if (mesh->HasUVs())
                uvs[uvIndex++] = mesh->GetUV(j);
                else
                uvs[uvIndex++] = UV(0.f, 0.f);
            }
        }
        
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] UVs buffer size: " << (sizeof(UV) * uvsCount / 1024) << "Kbytes" << endl;
        m_uvsBuff = new cl::Buffer(oclContext,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(UV) * uvsCount,
                                   uvs);
        deviceDesc->AllocMemory(m_uvsBuff->getInfo<CL_MEM_SIZE>());
        delete[] uvs;
        
        tEnd = WallClockTime();
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Texture maps translation time: " << int((tEnd - tStart) * 1000.0) << "ms" << endl;
    } else {
        m_texMapRGBBuff = NULL;
        m_texMapAlphaBuff = NULL;
        m_texMapDescBuff = NULL;
        m_meshTexsBuff = NULL;
        m_uvsBuff = NULL;
    }
    
    //--------------------------------------------------------------------------
	// Compile kernels
	//--------------------------------------------------------------------------
    
    size_t workGroupSize = 0;
    if (m_intersectionDevice->GetForceWorkGroupSize() > 0)
    workGroupSize = m_intersectionDevice->GetForceWorkGroupSize();

    // Set #define symbols
	stringstream ss;
	ss.precision(6);
	ss << scientific <<
    " -D PARAM_STARTLINE=" << startLine <<
    " -D PARAM_LIGHT_PATHS=" << m_lightPaths <<
    " -D PARAM_LIGHT_DEPTH=" << m_depth <<
    " -D PARAM_NLIGHTS=" << nLights <<
    " -D PARAM_PATH_COUNT=" << globalWorkGroupSize <<
    " -D PARAM_IMAGE_WIDTH=" << m_renderEngine->m_film->GetWidth() <<
    " -D PARAM_IMAGE_HEIGHT=" << m_renderEngine->m_film->GetHeight() <<
    " -D PARAM_RAY_EPSILON=" << RAY_EPSILON << "f" <<
    " -D PARAM_CLIP_YON=" << scene->m_camera->GetClipYon() << "f" <<
    " -D PARAM_CLIP_HITHER=" << scene->m_camera->GetClipHither() << "f" <<
    " -D PARAM_WORK_GROUP_SIZE=" << workGroupSize <<
    " -D PARAM_SEED=" << m_seed <<
    " -D PARAM_MAX_PATH_DEPTH=" << m_renderEngine->m_maxPathDepth <<
    " -D PARAM_RR_DEPTH=" << m_renderEngine->m_rrDepth <<
    " -D PARAM_RR_CAP=" << m_renderEngine->m_rrImportanceCap << "f" <<
    " -D PARAM_SAMPLE_PER_PIXEL=" << m_renderEngine->m_samplePerPixel
    ;

    if (enable_MAT_MATTE)
		ss << " -D PARAM_ENABLE_MAT_MATTE";
	if (enable_MAT_AREALIGHT)
		ss << " -D PARAM_ENABLE_MAT_AREALIGHT";
    
    if (m_texMapRGBBuff || m_texMapAlphaBuff)
        ss << " -D PARAM_HAS_TEXTUREMAPS";
    if (m_texMapAlphaBuff)
        ss << " -D PARAM_HAS_ALPHA_TEXTUREMAPS";
    
    if (scene->m_camera->m_lensRadius > 0.f) {
		ss <<
        " -D PARAM_CAMERA_HAS_DOF"
        " -D PARAM_CAMERA_LENS_RADIUS=" << scene->m_camera->m_lensRadius <<
        "f" << " -D PARAM_CAMERA_FOCAL_DISTANCE=" <<
        scene->m_camera->m_focalDistance << "f";
	}
    
    if (m_triLightsBuff) {
		ss <<
        " -D PARAM_DIRECT_LIGHT_SAMPLING" <<
        " -D PARAM_DL_LIGHT_COUNT=" << areaLightCount
        ;
	}
    
    m_cameraBuff = NULL;
    AppendMatrixDefinition(ss, "PARAM_RASTER2CAMERA", scene->m_camera->
                           GetRasterToCameraMatrix());
    AppendMatrixDefinition(ss, "PARAM_CAMERA2WORLD", scene->m_camera->
                           GetCameraToWorldMatrix());
    
    
    // Check if I have to recompile the kernels
	string newKernelParameters = ss.str();
	if (m_kernelsParameters != newKernelParameters) {
		m_kernelsParameters = newKernelParameters;
        
		// Compile sources
        std::vector<std::string> files;
        files.push_back("kernels/vplgpu_kernel.cl");
        cl::Program program = getProgram(oclContext, files);
		try {
			cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] Defined symbols: " << m_kernelsParameters << endl;
			cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] Compiling kernels " << endl;
            
			VECTOR_CLASS<cl::Device> buildDevice;
			buildDevice.push_back(oclDevice);
			program.build(buildDevice, m_kernelsParameters.c_str());
		} catch (cl::Error err) {
			cl::STRING_CLASS strError = program.getBuildInfo<
            CL_PROGRAM_BUILD_LOG>(oclDevice);
			cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] PathGPU compilation error:\n" << strError.c_str() << endl;
            
			throw err;
		}
        
        //----------------------------------------------------------------------
        // Init Integration kernel
        //----------------------------------------------------------------------
        
        delete m_initKernel;
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Compiling Init Kernel" << endl;
        m_initKernel = new cl::Kernel(program, "Init");
        
        if (m_intersectionDevice->GetForceWorkGroupSize() > 0) {
            m_initWorkGroupSize = m_intersectionDevice->GetForceWorkGroupSize();
            cerr << "[VPLGPURenderThread::" << m_threadIndex <<
            "] Forced work group size: " << m_initWorkGroupSize << endl;
        }
        
        delete m_integratorKernel;
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Compiling Integrator Kernel" << endl;
        m_integratorKernel = new cl::Kernel(program, "Integrator");
        

        
        //----------------------------------------------------------------------
		// InitFB kernel
		//----------------------------------------------------------------------
        
        delete m_initFBKernel;
		m_initFBKernel = new cl::Kernel(program, "InitFrameBuffer");
        
        m_initFBKernel->getWorkGroupInfo<size_t>(oclDevice, CL_KERNEL_WORK_GROUP_SIZE, &m_initFBWorkGroupSize);
		cerr << "[VPLGPURenderThread::" << m_threadIndex << "] Suggested work group size: " << m_initFBWorkGroupSize << endl;
        
        m_initFBWorkGroupSize = m_intersectionDevice->GetForceWorkGroupSize();
        cerr << "[VPLGPURenderThread::" << m_threadIndex <<
        "] Forced work group size: " << m_initFBWorkGroupSize << endl;
        
    }

    //--------------------------------------------------------------------------
	// Initialize
	//--------------------------------------------------------------------------
    
	// Clear the frame buffer
	cl::CommandQueue &oclQueue = m_intersectionDevice->GetOpenCLQueue();
	EnqueueInitFrameBufferKernel(true);
    
    // Initialize the path buffer
	m_initKernel->setArg(0, *m_pathsBuff);
	m_initKernel->setArg(1, *m_raysBuff);
    oclQueue.enqueueNDRangeKernel(*m_initKernel, cl::NullRange,
                                  cl::NDRange(globalWorkGroupSize),
                                  cl::NDRange(m_initWorkGroupSize));
    
    unsigned int argIndex = 0;
    
    m_integratorKernel->setArg(argIndex++, *m_pathsBuff);
    m_integratorKernel->setArg(argIndex++, *m_raysBuff);
    m_integratorKernel->setArg(argIndex++, *m_hitsBuff);
    m_integratorKernel->setArg(argIndex++, *m_vplsBuff);
    m_integratorKernel->setArg(argIndex++, *m_frameBufferBuff);
    m_integratorKernel->setArg(argIndex++, *m_materialsBuff);
    m_integratorKernel->setArg(argIndex++, *m_meshMatsBuff);
    m_integratorKernel->setArg(argIndex++, *m_meshIDBuff);
    m_integratorKernel->setArg(argIndex++, *m_colorsBuff);
    m_integratorKernel->setArg(argIndex++, *m_normalsBuff);
    m_integratorKernel->setArg(argIndex++, *m_trianglesBuff);
    m_integratorKernel->setArg(argIndex++, *m_triLightsBuff);
    m_integratorKernel->setArg(argIndex++, m_intersectionDevice->GetOpenCLBVH());
    m_integratorKernel->setArg(argIndex++, m_intersectionDevice->GetOpenCLVerts());
    // Textures
    if (m_texMapRGBBuff)
        m_integratorKernel->setArg(argIndex++, *m_texMapRGBBuff);
    if (m_texMapAlphaBuff)
        m_integratorKernel->setArg(argIndex++, *m_texMapAlphaBuff);
    if (m_texMapRGBBuff || m_texMapAlphaBuff) {
        m_integratorKernel->setArg(argIndex++, *m_texMapDescBuff);
        m_integratorKernel->setArg(argIndex++, *m_meshTexsBuff);
        m_integratorKernel->setArg(argIndex++, *m_uvsBuff);
    }
    
    // Reset statistics to be more accurate
	m_intersectionDevice->ResetPerformanceStats();
}

void VPLGPURenderThread::Interrupt() {
	if (m_renderThread)
		m_renderThread->interrupt();
}

void VPLGPURenderThread::Stop() {
	if (m_renderThread) {
		m_renderThread->interrupt();
		m_renderThread->join();
		delete m_renderThread;
		m_renderThread = NULL;
	}
    
	// Transfer of the frame buffer
	cl::CommandQueue &oclQueue = m_intersectionDevice->GetOpenCLQueue();
	const unsigned int pixelCount = m_renderEngine->m_film->GetWidth() * m_renderEngine->m_film->GetHeight();
	oclQueue.enqueueReadBuffer(
                               *m_frameBufferBuff,
                               CL_TRUE,
                               0,
                               sizeof(PixelGPU) * pixelCount,
                               m_frameBuffer);
    
	const OpenCLIntersectionDevice *deviceDesc = m_intersectionDevice;
	deviceDesc->FreeMemory(m_raysBuff->getInfo<CL_MEM_SIZE>());
	delete m_raysBuff;
	deviceDesc->FreeMemory(m_hitsBuff->getInfo<CL_MEM_SIZE>());
	delete m_hitsBuff;
	deviceDesc->FreeMemory(m_pathsBuff->getInfo<CL_MEM_SIZE>());
	delete m_pathsBuff;
	deviceDesc->FreeMemory(m_frameBufferBuff->getInfo<CL_MEM_SIZE>());
	delete m_frameBufferBuff;
	deviceDesc->FreeMemory(m_materialsBuff->getInfo<CL_MEM_SIZE>());
	delete m_materialsBuff;
	deviceDesc->FreeMemory(m_meshIDBuff->getInfo<CL_MEM_SIZE>());
	delete m_meshIDBuff;
	deviceDesc->FreeMemory(m_meshMatsBuff->getInfo<CL_MEM_SIZE>());
	delete m_meshMatsBuff;
	deviceDesc->FreeMemory(m_colorsBuff->getInfo<CL_MEM_SIZE>());
	delete m_colorsBuff;
	deviceDesc->FreeMemory(m_normalsBuff->getInfo<CL_MEM_SIZE>());
	delete m_normalsBuff;
	deviceDesc->FreeMemory(m_trianglesBuff->getInfo<CL_MEM_SIZE>());
	delete m_trianglesBuff;
	if (m_cameraBuff) {
		deviceDesc->FreeMemory(m_cameraBuff->getInfo<CL_MEM_SIZE>());
		delete m_cameraBuff;
	}
	if (m_triLightsBuff) {
		deviceDesc->FreeMemory(m_triLightsBuff->getInfo<CL_MEM_SIZE>());
		delete m_triLightsBuff;
	}
    
	m_started = false;
    
	// frameBuffer is delete on the destructor to allow image saving after
	// the rendering is finished
}

void VPLGPURenderThread::EnqueueInitFrameBufferKernel(const bool clearPBO) {
	cl::CommandQueue &oclQueue = m_intersectionDevice->GetOpenCLQueue();
    
	// Clear the frame buffer
    m_initFBKernel->setArg(0, *m_frameBufferBuff);
    oclQueue.enqueueNDRangeKernel(*m_initFBKernel, cl::NullRange,
                                  cl::NDRange(RoundUp<unsigned int>(
                                                                    m_renderEngine->m_film->GetWidth() *
                                                                    m_renderEngine->m_film->GetHeight(), m_initFBWorkGroupSize)),
                                  cl::NDRange(m_initFBWorkGroupSize));
    
}

void VPLGPURenderThread::RenderThreadImpl(VPLGPURenderThread *renderThread)
{
	cerr << "[VPLGPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread started" << endl;
    
    cl::Context &oclContext = renderThread->m_intersectionDevice->GetOpenCLContext();
	cl::CommandQueue &oclQueue = renderThread->m_intersectionDevice->GetOpenCLQueue();
	const unsigned int pixelCount = renderThread->m_renderEngine->m_film->GetWidth() * renderThread->m_renderEngine->m_film->GetHeight();
    const int globalWorkGroupSize = renderThread->m_renderEngine->m_globalWorkGroupSize;
    
    
    try {
        double startTime = WallClockTime();
        
        // This will be the intergration step once we have the VPLs
        // Rendering stage
        while (!boost::this_thread::interruption_requested()) {
            //if(renderThread->threadIndex == 0)
            //	cerr<< "[DEBUG] =================================" << endl;
            
            // Async. transfer of the frame buffer
            oclQueue.enqueueReadBuffer(
                                       *(renderThread->m_frameBufferBuff),
                                       CL_FALSE,
                                       0,
                                       sizeof(PixelGPU) * pixelCount,
                                       renderThread->m_frameBuffer);
            
            for(;;) {
                cl::Event event;
                for (unsigned int i = 0; i < 4; ++i) {
                    
                    // Trace eye rays and rays to cluster
                    //renderThread->m_intersectionDevice->
                    //EnqueueTraceRayBuffer(*(renderThread->m_raysBuff),
                    //                      *(renderThread->m_hitsBuff),
                    //                      globalWorkGroupSize);
                    
                    
                    // Integration Step
                    if (renderThread->m_triLightsBuff) {
                        // Only if direct light sampling is enabled
                        oclQueue.enqueueNDRangeKernel(*(renderThread->m_integratorKernel), cl::NullRange,
                                                      cl::NDRange(globalWorkGroupSize), cl::NDRange(renderThread->m_initWorkGroupSize),
                                                      NULL, (i == 0) ? &event : NULL);
                    }
                    
                }
                oclQueue.flush();
                
                event.wait();
                const double elapsedTime = WallClockTime() - startTime;
                
                //if(renderThread->threadIndex == 0)
                //	cerr<< "[DEBUG] =" << j << "="<< elapsedTime * 1000.0 << "ms" << endl;
                
                if ((elapsedTime > renderThread->m_renderEngine->m_screenRefreshInterval) ||
                    boost::this_thread::interruption_requested())
                break;
            }
            
            startTime = WallClockTime();
            
        }

		cerr << "[VPLGPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
	} catch (boost::thread_interrupted) {
		cerr << "[VPLGPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread halted" << endl;
	} catch (cl::Error err) {
		cerr << "[VPLGPURenderThread::" << renderThread->m_threadIndex << "] Rendering thread ERROR: " << err.what() << "(" << err.err() << ")" << endl;
	}

	oclQueue.enqueueReadBuffer(
                               *(renderThread->m_frameBufferBuff),
                               CL_TRUE,
                               0,
                               sizeof(PixelGPU) * pixelCount,
                               renderThread->m_frameBuffer);
}
