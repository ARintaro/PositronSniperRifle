#include "renderer.h"

#include "optix_stack_size.h"
#include "sutil/vec_math.h"

#include <cmath>

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

	renderParams.directLightCount = directLights.size();
	renderParams.deviceDirectLights = (DirectLightDescription*)deviceDirectLights.GetDevicePointer();

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

	outputBuffer->resize(newSize.x, newSize.y);

	renderParams.screenSize = newSize;

	SetCamera(curCameraSetting);
}

void PathTracer::AddMesh(std::shared_ptr<Mesh> mesh) {
	meshes.emplace_back(std::move(mesh));
}

void PathTracer::AddSphere(std::shared_ptr<Sphere> sphere) {
	spheres.emplace_back(std::move(sphere));
}

void PathTracer::AddCurve(std::shared_ptr<Curve> curve) {
	curves.emplace_back(std::move(curve));
}


void PathTracer::SetDirectLight(SceneObject& sceneObject) {
	sceneObject.isDirectLight = true;
	sceneObject.directLightId = directLights.size();

	directLights.push_back({});

	DirectLightDescription& desc = directLights.back();
	desc.id = sceneObject.directLightId;
	OptixAabb aabb = sceneObject.GetAabb();

	desc.vertex[0] = make_float3(aabb.minX, aabb.minY, aabb.minZ);
	desc.vertex[1] = make_float3(aabb.minX, aabb.minY, aabb.maxZ);
	desc.vertex[2] = make_float3(aabb.minX, aabb.maxY, aabb.minZ);
	desc.vertex[3] = make_float3(aabb.minX, aabb.maxY, aabb.maxZ);
	desc.vertex[4] = make_float3(aabb.maxX, aabb.minY, aabb.minZ);
	desc.vertex[5] = make_float3(aabb.maxX, aabb.minY, aabb.maxZ);
	desc.vertex[6] = make_float3(aabb.maxX, aabb.maxY, aabb.minZ);
	desc.vertex[7] = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);
}


Shader PathTracer::CreateShader(std::string name) {
	shaders.push_back({});
	Shader& shader = shaders.back();
	shader.name = name;
	shader.programIndex = shaders.size() - 1;
	return shader;
}

void PathTracer::SetCamera(const PathTracerCameraSetting& cameraSetting) {
	renderParams.frame.subframeCount = 0;

	curCameraSetting = cameraSetting;

	renderParams.camera.position = cameraSetting.position;

	renderParams.camera.direction = cameraSetting.focusDist * normalize(cameraSetting.lookAt - cameraSetting.position);

	const float tanFov = 2 * tan(cameraSetting.fov / 2 * M_PI / 180);

	const float aspect = (float)renderParams.screenSize.x / renderParams.screenSize.y;

	renderParams.camera.horizontal = cameraSetting.focusDist * tanFov * aspect * normalize(cross(renderParams.camera.direction, cameraSetting.up));
	renderParams.camera.vertical = cameraSetting.focusDist * tanFov * normalize(cross(renderParams.camera.direction, renderParams.camera.horizontal));
	renderParams.camera.lenRadius = cameraSetting.aperture / 2;
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

#ifdef _DEBUG
	optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

	optixPipelineCompileOptions = {};
	optixPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	
	// Object Type
	optixPipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE 
														| OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
														| OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
	// Turn Off Motion Blur
	optixPipelineCompileOptions.usesMotionBlur = false;

	// Important, Payload and AttributeNum
	optixPipelineCompileOptions.numPayloadValues = 2;
	optixPipelineCompileOptions.numAttributeValues = 3;
	
	// Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
#ifdef _DEBUG 
	optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_USER;
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
	// Raygen
	RaygenRecord raygenRecord;
	CheckOptiXErrors(optixSbtRecordPackHeader(raygenPrograms, &raygenRecord));
	raygenRecord.data = nullptr;
	
	raygenRecordsBuffer.Upload(&raygenRecord, 1);
	shaderBindingTable.raygenRecord = raygenRecordsBuffer.GetDevicePointer();

	// Miss
	MissRecord missRecord;
	CheckOptiXErrors(optixSbtRecordPackHeader(missPrograms, &missRecord));
	missRecord.data = nullptr;
	missRecordsBuffer.Upload(&missRecord, 1);
	shaderBindingTable.missRecordBase = missRecordsBuffer.GetDevicePointer();
	shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
	shaderBindingTable.missRecordCount = 1;

	
	// Shader
	vector<ShaderRecord> shaderRecords(shaders.size());
	for (auto& shader : shaders) {
		CheckOptiXErrors(optixSbtRecordPackHeader(shader.program, &shaderRecords[shader.programIndex]));
		shaderRecords[shader.programIndex].data = nullptr;
	}
	shaderRecordsBuffer.Upload(shaderRecords);

	shaderBindingTable.callablesRecordBase = shaderRecordsBuffer.GetDevicePointer();
	shaderBindingTable.callablesRecordCount = shaderRecords.size();
	shaderBindingTable.callablesRecordStrideInBytes = sizeof(ShaderRecord);


	// Hitgroup
	vector<HitgroupRecord> hitgroupRecords(meshes.size() + spheres.size() + curves.size());
	int sbtOffset = 0;
	for (int i = 0; i < meshes.size(); i++) {
		HitgroupRecord& record = hitgroupRecords[i + sbtOffset];
		meshes[i]->GetShaderBindingRecord(record, hitPrograms);
	}
	sbtOffset += geometryAccelDatas[MESH_OBJECT_TYPE].sbtRecordsCount;

	for (int i = 0; i < spheres.size(); i++) {
		HitgroupRecord& record = hitgroupRecords[i + sbtOffset];
		spheres[i]->GetShaderBindingRecord(record, hitPrograms);
	}
	sbtOffset += geometryAccelDatas[SPHERE_OBJECT_TYPE].sbtRecordsCount;

	for (int i = 0; i < curves.size(); i++) {
		HitgroupRecord& record = hitgroupRecords[i + sbtOffset];
		curves[i]->GetShaderBindingRecord(record, hitPrograms);
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
	// Build Curve
	BuildGeometryAccel(curves, geometryAccelDatas[CURVE_OBJECT_TYPE]);

	// Build IAS
	BuildInstanceAccel();

	// Solve Direct Lights
	deviceDirectLights.Upload(directLights);
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
	sbtOffset += geometryAccelDatas[SPHERE_OBJECT_TYPE].sbtRecordsCount;
	instanceId++;

	// Curve
	instances.push_back(OptixInstance{
		{1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0}, instanceId, sbtOffset, 255,
		flags, geometryAccelDatas[CURVE_OBJECT_TYPE].handle, {0, 0}
		});
	sbtOffset += geometryAccelDatas[CURVE_OBJECT_TYPE].sbtRecordsCount;
	instanceId++;

	
	instancesBuffer.Upload(instances);

	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = instancesBuffer.GetDevicePointer();
	buildInput.instanceArray.numInstances = static_cast<uint32_t>(instances.size());

	BuildAccel({ buildInput }, instancesAccelBuffer, iasHandle);
}

void PathTracer::BuildAccel(const vector<OptixBuildInput> inputs, CudaBuffer& structBuffer, OptixTraversableHandle& handle, unsigned int buildFlags) {
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | buildFlags;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	CheckOptiXErrors(
		optixAccelComputeMemoryUsage(
			optixDeviceContext,
			&accelOptions,
			inputs.data(),
			(int)inputs.size(),
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
			inputs.data(),
			(int)inputs.size(),
			tempBuffer.GetDevicePointer(),
			tempBuffer.GetSize(),
			outputBuffer.GetDevicePointer(),
			outputBuffer.GetSize(),
			&handle,
			&emitDesc,
			1
		)
	);

	CheckCudaErrors(
		cudaDeviceSynchronize()
	);

	uint64_t compactedSize;
	compactedSizeBuffer.Download(&compactedSize, 1);

	structBuffer.Alloc(compactedSize);

	CheckOptiXErrors(
		optixAccelCompact(
			optixDeviceContext,
			cudaStream,
			handle,
			structBuffer.GetDevicePointer(),
			structBuffer.GetSize(),
			&handle
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

		{
			// Curve
			OptixProgramGroupOptions programOptions = {};
			OptixProgramGroupDesc programDesc = {};
			programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			programDesc.hitgroup.moduleCH = optixModule;
			programDesc.hitgroup.entryFunctionNameCH = "__closesthit__curve";
			programDesc.hitgroup.moduleIS = optixModule;
			programDesc.hitgroup.entryFunctionNameIS = "__intersection__curve";
			CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &hitPrograms[CURVE_OBJECT_TYPE]));
		}

	}

	// Callables
	for (auto& shader : shaders) {
		OptixProgramGroupOptions programOptions = {};
		OptixProgramGroupDesc programDesc = {};
		programDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		programDesc.callables.moduleDC = optixModule;
		programDesc.callables.entryFunctionNameDC = shader.name.data();
		CheckOptiXErrorsLog(optixProgramGroupCreate(optixDeviceContext, &programDesc, 1, &programOptions, LOG, &LOG_SIZE, &shader.program));
	}
	
}

void PathTracer::CreateOptixPipeline() {
	vector<OptixProgramGroup> programs;
	programs.push_back(raygenPrograms);
	programs.push_back(missPrograms);

	for (auto& prog : hitPrograms) {
		programs.push_back(prog);
	}
	for (auto& shader : shaders) {
		programs.push_back(shader.program);
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

