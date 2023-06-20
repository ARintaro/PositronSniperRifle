#include "renderer.h"

#include "optix_stack_size.h"
#include "sutil/vec_math.h"

// This include may only appear in a single source file
// If you put it in a hearder file, there would be a link error
#include <optix_function_table_definition.h>
#include <sutil\sutil.h>


PathTracer::PathTracer() {

}

void PathTracer::Init() {
	std::clog << "Creating Path Tracer..." << std::endl;

	CreateOptixContext();

	CreateOptixModule();

	CreatePrograms();

	CreateOptixPipeline();

	BuildAllAccel();

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
	renderParams.traversable = iasHandle;
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
	meshes.emplace_back(std::move(mesh));
}

void PathTracer::AddSphere(Sphere&& sphere) {
	spheres.emplace_back(std::move(sphere));
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
	
	// Object Type
	optixPipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE 
														| OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
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
	
	// Get Builtin Sphere IS Module
	OptixBuiltinISOptions builtinISOptions = {};

	builtinISOptions.usesMotionBlur = false;
	builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
	CheckOptiXErrorsLog(
		optixBuiltinISModuleGet(optixDeviceContext, &optixModuleCompileOptions, &optixPipelineCompileOptions, &builtinISOptions, &optixSphereISModule)
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

	vector<HitgroupRecord> hitgroupRecords(meshes.size() + spheres.size());

	int sbtOffset = 0;
	for (int i = 0; i < meshes.size(); i++) {
		HitgroupRecord& record = hitgroupRecords[i + sbtOffset];
		meshes[i].GetShaderBindingRecord(record, hitPrograms);
	}

	sbtOffset += meshes.size();
	for (int i = 0; i < spheres.size(); i++) {
		HitgroupRecord& record = hitgroupRecords[i + sbtOffset];
		spheres[i].GetShaderBindingRecord(record, hitPrograms);
	}

	hitRecordsBuffer.Upload(hitgroupRecords);
	shaderBindingTable.hitgroupRecordBase = hitRecordsBuffer.GetDevicePointer();
	shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	shaderBindingTable.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void PathTracer::BuildAllAccel() {
	geometryAccelDatas.resize(OBJECT_TYPE_COUNT);
	// Build Mesh
	BuildGeometryAccel(meshes, geometryAccelDatas[MESH_OBJECT_TYPE]);
	// Build Sphere
	BuildGeometryAccel(spheres, geometryAccelDatas[SPHERE_OBJECT_TYPE]);

	// Build IAS
	BuildInstanceAccel();
}

void PathTracer::BuildInstanceAccel() {
	vector<OptixInstance> instances;
	uint32_t flags = OPTIX_INSTANCE_FLAG_NONE;

	uint32_t sbtOffset = 0;
	uint32_t instanceId = 0;

	// Mesh
	instances.emplace_back(OptixInstance{
		{1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0}, instanceId, sbtOffset, 255,
		flags, geometryAccelDatas[MESH_OBJECT_TYPE].handle, {0, 0}
		});
	sbtOffset += geometryAccelDatas[MESH_OBJECT_TYPE].sbtRecordsCount;
	instanceId++;

	// Sphere
	instances.push_back(OptixInstance{
		{1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0}, instanceId, sbtOffset, 255,
		flags, geometryAccelDatas[SPHERE_OBJECT_TYPE].handle, {0, 0}
	});
	
	instancesBuffer.Upload(instances);

	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = instancesBuffer.GetDevicePointer();
	buildInput.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

	BuildAccel({ buildInput }, instancesAccelBuffer, iasHandle);
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

		hitPrograms.resize(OBJECT_TYPE_COUNT);

		{
			// Mesh
			OptixProgramGroupOptions programOptions = {};
			OptixProgramGroupDesc programDesc = {};
			programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			programDesc.hitgroup.moduleCH = optixModule;
			programDesc.hitgroup.entryFunctionNameCH = "__closesthit__mesh";
			CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &hitPrograms[MESH_OBJECT_TYPE]));
		}

		{
			// Sphere
			OptixProgramGroupOptions programOptions = {};
			OptixProgramGroupDesc programDesc = {};
			programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			programDesc.hitgroup.moduleCH = optixModule;
			programDesc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
			programDesc.hitgroup.moduleIS = optixSphereISModule;
			CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &hitPrograms[SPHERE_OBJECT_TYPE]));
		}

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

