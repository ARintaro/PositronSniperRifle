

#include "renderParams.h"
#include "sutil\vec_math.h"

#include "random.h"

#include <crt\host_defines.h>
#include <optix_device.h>
#include "helpers.h"


extern "C" __constant__ RenderParams renderParams;

struct TraceResult {
    int missed = 0;

    float3 position;
    float3 normal;

    Material material;
};


static __forceinline__ __device__
void* UnpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  PackPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* GetPerRayData()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(UnpackPointer(u0, u1));
}

static __forceinline__ __device__ 
float3 RandomInUnitSphere(unsigned int& seed) {
    while (true) {
        float3 v = make_float3(rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f);
        if (dot(v, v) >= 1.0f) continue;
        return v;
    }
}

static __forceinline__ __device__ 
float3 RandomSampleHemisphere(unsigned int& seed, const float3& normal) {
    const float3 vec_in_sphere = RandomInUnitSphere(seed);
    if (dot(vec_in_sphere, normal) > 0.0f)
        return vec_in_sphere;
    else
        return -vec_in_sphere;
}

static __forceinline__ __device__ float FresnelSchlick(float inCosine, float f0) {
    float oneMinusCos = 1.0f - inCosine;
    float oneMinusCosSqr = oneMinusCos * oneMinusCos;
    float fresnel = f0 + (1.0f - f0) * oneMinusCosSqr * oneMinusCosSqr * oneMinusCos;
    return fresnel;
}

static __forceinline__ __device__ float3 Refract(const float3& in, const float3& normal, float refractivity) {
    auto cos_theta = fminf(dot(-in, normal), 1.0f);
    float3 r_out_perp = refractivity * (in + cos_theta * normal);
    float3 r_out_parallel = -sqrtf(fabs(1.0f - dot(r_out_perp, r_out_perp))) * normal;
    return r_out_perp + r_out_parallel;
}

static __forceinline__ __device__
void RayTrace(float3 position, float3 rayDir, int rayType, TraceResult* result) {
    uint32_t u0, u1;
    PackPointer(result, u0, u1);
    optixTrace(
        renderParams.traversable,
        position,
        rayDir,
        0.001, 1e20f, 0,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        rayType,
        RAY_TYPE_COUNT,
        rayType, u0, u1);
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

        float3 rayOrigin = camera.position;
        float3 rayDir = normalize(camera.direction + (screenPos.y - 0.5f) * camera.vertical + (screenPos.x - 0.5f) * camera.horizontal);

        float3 attenuation = make_float3(1.f);

        for (int depth = 0; depth < renderParams.maxDepth; depth++) {
            traceResult.missed = true;

            if (rnd(seed) > renderParams.russianRouletteProbability) {
                break;
            }

            RayTrace(rayOrigin, rayDir, RADIANCE_RAY_TYPE, &traceResult);

            if (traceResult.missed) {
                break;
            }

            optixDirectCall<void, unsigned int&, float3&, TraceResult&, float3&, float3&, float3&>(traceResult.material.programIndex, seed, result,traceResult, attenuation, rayOrigin, rayDir);
        }

    }

    result /= renderParams.samplesPerLaunch;

    const uint32_t colorBufferIndex = (renderParams.screenSize.x - px - 1) + (renderParams.screenSize.y - py - 1) * renderParams.screenSize.x;

    const int subframeCount = renderParams.frame.subframeCount;

    if (subframeCount > 0) {
        result = lerp(renderParams.frame.colorBuffer[colorBufferIndex], result, 1. / subframeCount);
    }

    // const uint32_t colorBufferIndex = px + py * renderParams.screenSize.x;
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
    const float3& A = data.vertex[index.x];
    const float3& B = data.vertex[index.y];
    const float3& C = data.vertex[index.z];
    const float3 normal = normalize(cross(B - A, C - A));
    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

    const float3 rayDir = optixGetWorldRayDirection();

    result.missed = false;
    result.normal = -normal;
    result.position = position;
    result.material = material;
}


extern "C" __global__ void __closesthit__sphere() {
    TraceResult& result = *GetPerRayData<TraceResult>();

    const ShaderBindingData& sbtData = *(const ShaderBindingData*)optixGetSbtDataPointer();
    const Material& material = sbtData.material;
    const DeviceSphereData& data = sbtData.data.sphere;

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 normal = normalize(position - data.position);

    const float3 rayDir = optixGetWorldRayDirection();

    result.missed = false;
    result.normal = normal;
    result.position = position;
    result.material = material;    
}


extern "C" __device__ void __direct_callable__naive_diffuse(unsigned int& seed, float3& result, TraceResult& traceResult, float3& attenuation, float3& rayOrigin, float3& rayDir) {
    NaviveDiffuseData& data = *(NaviveDiffuseData*)traceResult.material.data;

    result += data.emission * attenuation;

    rayOrigin = traceResult.position;
    rayDir = RandomSampleHemisphere(seed, traceResult.normal);

    float cosine = dot(rayDir, traceResult.normal);

    attenuation *= 2 * cosine * data.albedo / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__naive_mirror(unsigned int& seed, float3 & result, TraceResult & traceResult, float3 & attenuation, float3 & rayOrigin, float3 & rayDir) {
    NaiveMirrorData& data = *(NaiveMirrorData*)traceResult.material.data;

    rayOrigin = traceResult.position;
    rayDir = reflect(rayDir, traceResult.normal);

    attenuation *= 1 / renderParams.russianRouletteProbability;
}