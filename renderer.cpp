#include "renderer.h"

#include "optix_stack_size.h"
#include "sutil/vec_math.h"

// This include may only appear in a single source file
// If you put it in a hearder file, there would be a link error
#include <optix_function_table_definition.h>
#include <sutil\sutil.h>

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

	T data;
};

using RaygenRecord = SbtRecord<void*>;
using MissRecord = SbtRecord<void*>;
using HitgroupRecord = SbtRecord<MeshShaderBindingData>;


PathTracer::PathTracer() {

}

void PathTracer::Init() {
	std::clog << "Creating Path Tracer..." << std::endl;

	CreateOptixContext();

	CreateOptixModule();

	CreatePrograms();

	CreateOptixPipeline();

	BuildAccel();

	BuildShaderBindingTable();

	renderParamsBuffer.Alloc(sizeof(renderParams));

	outputBuffer = new sutil::CUDAOutputBuffer<float3>(sutil::CUDAOutputBufferType::GL_INTEROP, 512, 512);

	outputBuffer->setStream(cudaStream);

	std::clog << "Create Path Tracer Successfully !" << std::endl;
}

void PathTracer::Render() {
	if (renderParams.screenSize.x == 0 || renderParams.screenSize.y == 0) {
		return;
	}
	
	renderParams.frame.frameId++;
	renderParams.traversable = accelHandle;
	renderParams.frame.subframeCount++;
	renderParams.frame.colorBuffer = outputBuffer->map();

	renderParamsBuffer.Upload(&renderParams, 1);

	CheckOptiXErrors(
		optixLaunch(
			optixPipeline,
			cudaStream,
			renderParamsBuffer.GetDevicePointer(),
			renderParamsBuffer.GetSize(),
			&shaderBindingTable,
			renderParams.screenSize.x,
			renderParams.screenSize.y,
			1
		)
	);

	outputBuffer->unmap();

	CheckCudaErrors(
		cudaDeviceSynchronize()
	);
}

void PathTracer::Resize(const int2& newSize) {
	if (newSize.x == 0 || newSize.y == 0) return;

	// colorBuffer.Resize(newSize.x * newSize.y * sizeof(Color));

	outputBuffer->resize(newSize.x, newSize.y);

	renderParams.screenSize = newSize;

	SetCamera(curCameraSetting);
}


void PathTracer::AddMesh(Mesh&& mesh) {
	meshes.push_back(std::move(mesh));
}

void PathTracer::SetCamera(const PathTracerCameraSetting& cameraSetting) {
	renderParams.frame.subframeCount = 0;

	curCameraSetting = cameraSetting;

	renderParams.camera.position = cameraSetting.position;
	renderParams.camera.direction = normalize(cameraSetting.lookAt - cameraSetting.position);

	const float tanFov = 2 * tan(cameraSetting.fov / 2 * M_PI / 180);

	const float aspect = (float)renderParams.screenSize.x / renderParams.screenSize.y;

	renderParams.camera.horizontal = tanFov * aspect * normalize(cross(renderParams.camera.direction, cameraSetting.up));
	renderParams.camera.vertical = tanFov * normalize(cross(renderParams.camera.direction, renderParams.camera.horizontal));
}

int2 PathTracer::GetScreenSize() {
	return renderParams.screenSize;
}

void PathTracer::SaveImage(const char* fileName) {
	sutil::ImageBuffer buffer;
	buffer.data = outputBuffer->getHostPointer();
	buffer.width = outputBuffer->width();
	buffer.height = outputBuffer->height();
	buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

	

	sutil::saveImage(fileName, buffer, true);
}



static void PrintOptixLog(uint32_t level, const char* tag, const char* msg, void* /* callback_data */) {
	std::clog << "[Optix Log]" << "[" << level << "]" << " " << msg << std::endl;
}

void PathTracer::CreateOptixContext() {
	// TODO : Config
	const int cudaDeviceId = 0;

	CheckCudaErrors(cudaSetDevice(cudaDeviceId));
	CheckCudaErrors(cudaStreamCreate(&cudaStream));

	cudaGetDeviceProperties(&deviceProps, cudaDeviceId);

	std::clog << "Running on device " << deviceProps.name << std::endl;

	// zero means take the current context
	cudaContext = 0;

	OptixDeviceContextOptions optixOptions = { };

	optixOptions.logCallbackFunction = &PrintOptixLog;
	optixOptions.logCallbackLevel = 4;

	CheckOptiXErrors(optixDeviceContextCreate(cudaContext, &optixOptions, &optixDeviceContext));
}

void PathTracer::CreateOptixModule() {
	optixModuleCompileOptions = {};

	optixModuleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

#if !defined( NDEBUG )
	optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

	optixPipelineCompileOptions = {};
	optixPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	
	// Turn Off Motion Blur
	optixPipelineCompileOptions.usesMotionBlur = false;

	// Important, Payload and AttributeNum
	optixPipelineCompileOptions.numPayloadValues = 2;
	optixPipelineCompileOptions.numAttributeValues = 2;
	
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
	optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
	optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif

	optixPipelineCompileOptions.pipelineLaunchParamsVariableName = "renderParams";
	

	size_t      inputSize = 0;
	const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "devicePrograms.cu", inputSize);

	CheckOptiXErrorsLog(
		optixModuleCreate(
			optixDeviceContext,
			&optixModuleCompileOptions, &optixPipelineCompileOptions, input, inputSize, LOG, &LOG_SIZE, &optixModule)
	);
	
	
}

void PathTracer::BuildShaderBindingTable() {

	RaygenRecord raygenRecord;
	CheckOptiXErrors(optixSbtRecordPackHeader(raygenPrograms, &raygenRecord));
	raygenRecord.data = nullptr;
	
	raygenRecordsBuffer.Upload(&raygenRecord, 1);
	shaderBindingTable.raygenRecord = raygenRecordsBuffer.GetDevicePointer();


	MissRecord missRecord;
	CheckOptiXErrors(optixSbtRecordPackHeader(missPrograms, &missRecord));
	missRecord.data = nullptr;
	missRecordsBuffer.Upload(&missRecord, 1);
	shaderBindingTable.missRecordBase = missRecordsBuffer.GetDevicePointer();
	shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
	shaderBindingTable.missRecordCount = 1;

	vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < meshes.size(); i++) {
		int objectType = 0;

		HitgroupRecord record;
		CheckOptiXErrors(optixSbtRecordPackHeader(hitPrograms[objectType], &record));
		record.data.color = meshes[i].color;
		record.data.vertex = (float3*)vertexBuffers[i].GetDevicePointer();
		record.data.index = (int3*)indexBuffers[i].GetDevicePointer();
		record.data.material = meshes[i].material;

		hitgroupRecords.push_back(record);
	}

	hitRecordsBuffer.Upload(hitgroupRecords);
	shaderBindingTable.hitgroupRecordBase = hitRecordsBuffer.GetDevicePointer();
	shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	shaderBindingTable.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void PathTracer::BuildAccel() {
	vertexBuffers.resize(meshes.size());
	indexBuffers.resize(meshes.size());

	accelHandle = 0;

	vector<OptixBuildInput> triangleInput(meshes.size());
	vector<CUdeviceptr> deviceVertices(meshes.size());
	vector<CUdeviceptr> deviceIndices(meshes.size());
	vector<uint32_t> triangleInputFlags(meshes.size());

	for (int meshId = 0; meshId < meshes.size(); meshId++) {
		Mesh& mesh = meshes[meshId];

		OptixBuildInput& input = triangleInput[meshId];

		vertexBuffers[meshId].Upload(mesh.vertex);
		indexBuffers[meshId].Upload(mesh.index);

		input = {};
		input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		deviceVertices[meshId] = vertexBuffers[meshId].GetDevicePointer();
		deviceIndices[meshId] = indexBuffers[meshId].GetDevicePointer();

		OptixBuildInputTriangleArray& triangleArray = input.triangleArray;

		triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleArray.vertexStrideInBytes = sizeof(float3);
		triangleArray.numVertices = (int)mesh.vertex.size();
		triangleArray.vertexBuffers = &deviceVertices[meshId];

		triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleArray.indexStrideInBytes = sizeof(int3);
		triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleArray.indexBuffer = deviceIndices[meshId];

		triangleInputFlags[meshId] = 0;

		// SBT Setting
		triangleArray.flags = &triangleInputFlags[meshId];
		triangleArray.numSbtRecords = 1;
		triangleArray.sbtIndexOffsetBuffer = 0;
		triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	CheckOptiXErrors(
		optixAccelComputeMemoryUsage(
			optixDeviceContext,
			&accelOptions,
			triangleInput.data(),
			(int)triangleInput.size(),
			&blasBufferSizes
		)
	);

	CudaBuffer compactedSizeBuffer;
	compactedSizeBuffer.Alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.GetDevicePointer();


	CudaBuffer tempBuffer;
	tempBuffer.Alloc(blasBufferSizes.tempSizeInBytes);

	CudaBuffer outputBuffer;
	outputBuffer.Alloc(blasBufferSizes.outputSizeInBytes);

	CheckOptiXErrors(
		optixAccelBuild(
			optixDeviceContext,
			cudaStream,
			&accelOptions,
			triangleInput.data(),
			(int)triangleInput.size(),
			tempBuffer.GetDevicePointer(),
			tempBuffer.GetSize(),
			outputBuffer.GetDevicePointer(),
			outputBuffer.GetSize(),
			&accelHandle,
			&emitDesc,
			1
		)
	);

	CheckCudaErrors(
		cudaDeviceSynchronize()
	);

	uint64_t compactedSize;
	compactedSizeBuffer.Download(&compactedSize, 1);

	accelStructBuffer.Alloc(compactedSize);
	
	CheckOptiXErrors(
		optixAccelCompact(
			optixDeviceContext,
			cudaStream,
			accelHandle,
			accelStructBuffer.GetDevicePointer(),
			accelStructBuffer.GetSize(),
			&accelHandle
		)
	);

	CheckCudaErrors(
		cudaDeviceSynchronize()
	);

	outputBuffer.Free();
	tempBuffer.Free();
	compactedSizeBuffer.Free();
}




void PathTracer::CreatePrograms() {

	{
		// Raygen
		OptixProgramGroupOptions programOptions = {};
		OptixProgramGroupDesc programDesc = {};
		programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		programDesc.raygen.module = optixModule;
		programDesc.raygen.entryFunctionName = "__raygen__renderFrame";

		CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &raygenPrograms));
	}

	{
		// Miss
		OptixProgramGroupOptions programOptions = {};
		OptixProgramGroupDesc programDesc = {};
		programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		programDesc.miss.module = optixModule;
		programDesc.miss.entryFunctionName = "__miss__radiance";

		CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &missPrograms));
	}

	{
		// Hitgroup

		hitPrograms.resize(1);
		OptixProgramGroupOptions programOptions = {};
		OptixProgramGroupDesc programDesc = {};
		programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		programDesc.hitgroup.moduleCH = optixModule;
		programDesc.hitgroup.moduleAH = optixModule;
		programDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		programDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &hitPrograms[0]));
	}
	
}

void PathTracer::CreateOptixPipeline() {
	vector<OptixProgramGroup> programs;
	programs.push_back(raygenPrograms);
	programs.push_back(missPrograms);

	for (auto prog : hitPrograms) {
		programs.push_back(prog);
	}

	optixPipelineLinkOptions.maxTraceDepth = 2;

	CheckOptiXErrorsLog(
		optixPipelineCreate(
			optixDeviceContext,
			&optixPipelineCompileOptions,
			&optixPipelineLinkOptions,
			programs.data(),
			programs.size(),
			LOG,
			&LOG_SIZE,
			&optixPipeline
		)
	);

	OptixStackSizes optixStackSizes = {};

	for (auto prog : programs) {
		optixUtilAccumulateStackSizes(prog, &optixStackSizes, optixPipeline);
	}

	// TODO : Config

	const uint32_t maxContinuationCallableDepth = 0;
	const uint32_t maxDirectCallableDepth = 3;
	uint32_t directCallableStackSizeFromTraversable;
	uint32_t directCallableStackSizeFromState;
	uint32_t continuationStackSize;

	CheckOptiXErrors(
		optixUtilComputeStackSizes(
			&optixStackSizes,
			optixPipelineLinkOptions.maxTraceDepth,
			maxContinuationCallableDepth,
			maxDirectCallableDepth,
			&directCallableStackSizeFromTraversable,
			&directCallableStackSizeFromState,
			&continuationStackSize
		)
	);

	const uint32_t maxTraversalDepth = 2;

	CheckOptiXErrors(
		optixPipelineSetStackSize(
			optixPipeline,
			directCallableStackSizeFromTraversable,
			directCallableStackSizeFromState,
			continuationStackSize,
			maxTraversalDepth
		)
	);

}

