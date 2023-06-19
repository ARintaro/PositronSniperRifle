#pragma once

#include <vector>
#include "cudaBuffer.h"
#include "renderParams.h"
#include "mesh.h"
#include <sutil/CUDAOutputBuffer.h>

using std::vector;

struct PathTracerCameraSetting {
	float3 position;
	float3 lookAt;
	float3 up;
	float fov;
};



class PathTracer {
public:
	PathTracer();

	void Init();

	void Render();

	void Resize(const int2& newSize);

	void AddMesh(Mesh&& mesh);

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

	void BuildAccel();

protected:
	vector<Mesh> meshes;
	vector<CudaBuffer> vertexBuffers;
	vector<CudaBuffer> indexBuffers;

	PathTracerCameraSetting curCameraSetting;


	CUcontext cudaContext;
	CUstream cudaStream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixDeviceContext;

	OptixPipeline optixPipeline;
	OptixPipelineCompileOptions optixPipelineCompileOptions;
	OptixPipelineLinkOptions optixPipelineLinkOptions;

	OptixModule optixModule;
	OptixModuleCompileOptions optixModuleCompileOptions;

	OptixProgramGroup raygenPrograms;
	CudaBuffer raygenRecordsBuffer;

	OptixProgramGroup missPrograms;
	CudaBuffer missRecordsBuffer;

	vector<OptixProgramGroup> hitPrograms;
	CudaBuffer hitRecordsBuffer;

	OptixTraversableHandle accelHandle;

	OptixShaderBindingTable shaderBindingTable = {};

	RenderParams renderParams;

	CudaBuffer renderParamsBuffer;

	CudaBuffer accelStructBuffer;
};