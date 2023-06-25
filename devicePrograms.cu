

#include "renderParams.h"
#include "random.h"
#include "disney.cuh"
#include "bezier.cuh"
#include "material.cuh"
#include "deviceHelper.cuh"


static __forceinline__ __device__
bool ApplyAirScatter(float3& result, PathState& state, TraceResult& traceResult) {
    if (renderParams.globalFogDensity <= 0) return false;

    float scatterProbability = 1 - expf(-renderParams.globalFogDensity * traceResult.distance);
    // float scatterProbability = min(renderParams.globalFogDensity * distance, 1.f);
    float sampledScatter = rnd(state.seed);

    if (sampledScatter < scatterProbability) {
        // 发生散射
        float scateredDistance = -(1.f / renderParams.globalFogDensity) * logf(1 - sampledScatter);

        // 计算trace结果
        traceResult.directLightId = -1;
        traceResult.distance = scateredDistance;
        traceResult.position = state.rayOrigin + scateredDistance * state.rayDir;
        traceResult.normal = RandomInUnitSphere(state.seed);

        SAMPLE_DIRECT_LIGHT(result, state, traceResult, renderParams.globalFogAttenuation / scatterProbability);

        state.rayOrigin = traceResult.position;
        state.rayDir = reflect(state.rayDir, traceResult.normal);
        state.attenuation *= renderParams.globalFogAttenuation / renderParams.russianRouletteProbability;
        return true;
    }

    return false;
}

extern "C" __global__ void __raygen__renderFrame() {
    const int px = optixGetLaunchIndex().x;
    const int py = optixGetLaunchIndex().y;

    const auto& camera = renderParams.camera;

    unsigned int seed = tea<4>(px + py * renderParams.screenSize.x, renderParams.frame.frameId);

    float3 result = make_float3(0);

    TraceResult traceResult;

    for (int i = 0; i < renderParams.samplesPerLaunch; i++) {
        const float2 screenPos = make_float2(px + rnd(seed), py + rnd(seed)) / make_float2(renderParams.screenSize);

        float2 unitDisk = camera.lenRadius * RandomInUnitDisk(seed);

        PathState state;
        state.seed = seed;
        state.rayOrigin = camera.position + unitDisk.x * normalize(camera.horizontal) + unitDisk.y * normalize(camera.vertical);
        state.rayDir = normalize(camera.direction + (screenPos.y - 0.5f) * camera.vertical + (screenPos.x - 0.5f) * camera.horizontal);
        state.attenuation = make_float3(1.f);
        state.supposedColor = make_float3(1.f);
        state.collectDirectLight = true;
       

        for (int depth = 0; depth < renderParams.maxDepth; depth++) {
            traceResult.missed = true;
            traceResult.distance = 1e9;

            if (rnd(seed) > renderParams.russianRouletteProbability) {
                break;
            }

            RayTrace(state.rayOrigin, state.rayDir, RADIANCE_RAY_TYPE, &traceResult);

            if (ApplyAirScatter(result, state, traceResult)) continue;

            if (traceResult.missed) break;
  
            optixDirectCall<void, float3&, PathState&, TraceResult&>(traceResult.material.programIndex, result, state, traceResult);

            if (traceResult.missed) break;
        }

    }

    result /= renderParams.samplesPerLaunch;
    const uint32_t colorBufferIndex = (renderParams.screenSize.x - px - 1) + (renderParams.screenSize.y - py - 1) * renderParams.screenSize.x;
    const int subframeCount = renderParams.frame.subframeCount;

    if (isnan(result.x)) result.x = 0;
    if (isnan(result.y)) result.y = 0;
    if (isnan(result.z)) result.z = 0;

    if (subframeCount > 0) {
        result = lerp(renderParams.frame.colorBuffer[colorBufferIndex], result, 1. / subframeCount);
    }

    renderParams.frame.colorBuffer[colorBufferIndex] = result;
}

extern "C" __global__ void __miss__radiance() {

}

extern "C" __global__ void __closesthit__mesh() { 
    TraceResult &result = *GetPerRayData<TraceResult>();

    const ShaderBindingData& sbtData = *(const ShaderBindingData*)optixGetSbtDataPointer();
    const Material& material = sbtData.material;
    const DeviceMeshData& data = sbtData.data.mesh;

    const int primID = optixGetPrimitiveIndex();
    const int3 index = data.index[primID];
    float3 normal = make_float3(0);
    float2 bary = optixGetTriangleBarycentrics();
    float2 texcoord = make_float2(0);
    float3 tangent = make_float3(0);

    if (data.normal) {
        normal = (1.f - bary.x - bary.y) * data.normal[index.x] + bary.x * data.normal[index.y] + bary.y * data.normal[index.z];
    } else {
        const float3& A = data.vertex[index.x];
        const float3& B = data.vertex[index.y];
        const float3& C = data.vertex[index.z];
        normal = normalize(cross(B - A, C - A));
    }

    if (data.texcoord) {
        texcoord = (1.f - bary.x - bary.y) * data.texcoord[index.x] + bary.x * data.texcoord[index.y] + bary.y * data.texcoord[index.z];

        {
            // tangent
            float3 e1 = data.vertex[index.y] - data.vertex[index.x], e2 = data.vertex[index.z] - data.vertex[index.x];
            float2 duv1 = data.texcoord[index.y] - data.texcoord[index.x], duv2 = data.texcoord[index.z] - data.texcoord[index.x];
            tangent = (duv1.y * e2 - duv2.y * e1) / (duv1.y * duv2.x - duv2.y * duv1.x);
            tangent = normalize(tangent - dot(tangent, normal) * normal);
        }
    }

    

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, normal) > 0) {
        normal = -normal;
        result.outer = false;
    }

    result.missed = false;
    result.normal = normal;
    result.position = position;
    result.material = material;
    result.distance = optixGetRayTmax();
    result.directLightId = sbtData.directLightId;
    result.texcoord = texcoord;
    result.tangent = tangent;
}


extern "C" __global__ void __closesthit__sphere() {
    TraceResult& result = *GetPerRayData<TraceResult>();

    const ShaderBindingData& sbtData = *(const ShaderBindingData*)optixGetSbtDataPointer();
    const Material& material = sbtData.material;
    const DeviceSphereData& data = sbtData.data.sphere;

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float3 normal = normalize(position - data.position);

    const float3 rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, normal) > 0) {
        normal = -normal;
        result.outer = false;
    }

    result.missed = false;
    result.normal = normal;
    result.position = position;
    result.material = material;    
    result.distance = optixGetRayTmax();
    result.directLightId = -1;
    // TODO : texcrood

    // TODO : tangent and bitangent
}

