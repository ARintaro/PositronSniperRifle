#pragma once

#include <vector>
#include "cudaBuffer.h"
#include "renderParams.h"
#include "mesh.h"
#include <sutil/CUDAOutputBuffer.h>
#include "shader.hpp"

using std::vector;

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	void* data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	ShaderBindingData data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) ShaderRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	void* data;
};


struct PathTracerCameraSetting {
	float3 position;
	float3 lookAt;
	float3 up;
	float fov;
	float aperture = 0.03;
	float focusDist = 2;
};


struct GeometryAccelData {
	OptixTraversableHandle handle = 0;
	CudaBuffer structBuffer;
	uint32_t sbtRecordsCount;
};


class PathTracer {
public:
	PathTracer();

	void Init();

	void Render();

	void Resize(const int2& newSize);

	void AddMesh(Mesh&& mesh);

	void AddSphere(Sphere&& sphere);

	void AddCurve(Curve&& curve);

	Shader CreateShader(std::string name);

	void SetCamera(const PathTracerCameraSetting& cameraSetting);

	int2 GetScreenSize();

	void SaveImage(const char* fileName);

	sutil::CUDAOutputBuffer<float3>* outputBuffer;
 
protected:

	void CreateOptixContext();

	void CreateOptixModule();

	void CreatePrograms();

	void CreateOptixPipeline();

	void BuildShaderBindingTable();

	void BuildAllAccel();

	void BuildInstanceAccel();

protected:
	// Scene Inputs
	vector<Mesh> meshes;
	vector<Sphere> spheres;
	vector<Curve> curves;

	PathTracerCameraSetting curCameraSetting;

	// Render Params
	RenderParams renderParams;
	CudaBuffer renderParamsBuffer;

	// Programs
	OptixProgramGroup raygenPrograms;
	CudaBuffer raygenRecordsBuffer;

	OptixProgramGroup missPrograms;
	CudaBuffer missRecordsBuffer;

	vector<OptixProgramGroup> hitPrograms;
	CudaBuffer hitRecordsBuffer;

	vector<Shader> shaders;
	CudaBuffer shaderRecordsBuffer;

	// Optix 
	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixDeviceContext;
	OptixPipeline optixPipeline;
	OptixPipelineCompileOptions optixPipelineCompileOptions;
	OptixPipelineLinkOptions optixPipelineLinkOptions;

	OptixModule optixModule;
	OptixModule optixSphereISModule;
	OptixModuleCompileOptions optixModuleCompileOptions;

	vector<GeometryAccelData> geometryAccelDatas;
	OptixShaderBindingTable shaderBindingTable = {};

	CudaBuffer instancesBuffer;
	CudaBuffer instancesAccelBuffer;
	OptixTraversableHandle iasHandle;

protected:
	void BuildAccel(const vector<OptixBuildInput> inputs, CudaBuffer& structBuffer, OptixTraversableHandle& handle, unsigned int buildFlags = 0);

	template<typename T>
	void BuildGeometryAccel(vector<T>& objects, GeometryAccelData& data, unsigned int buildFlags = 0) {
		// One Record Per Object
		data.sbtRecordsCount = objects.size();

		if (objects.empty()) {
			return;
		}

		vector<OptixBuildInput> inputs(objects.size());

		for (int objectId = 0; objectId < objects.size(); objectId++) {
			objects[objectId].GetBuildInput(inputs[objectId]);
		}

		BuildAccel(inputs, data.structBuffer, data.handle);
	}
};