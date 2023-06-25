#pragma once

#include "renderParams.h"
#include <crt\host_defines.h>
#include <optix_device.h>
#include "sutil\vec_math.h"
#include <random.h>
#include <cassert>

extern "C" __constant__ RenderParams renderParams;

__constant__ float MPI = M_PI;
__constant__ float INV_PI = 1 / M_PI;
__constant__ float INV_2PI = 1 / (2 * M_PI);


struct PathState {
    unsigned int seed;
    bool collectDirectLight;
    float3 attenuation;
    float3 supposedColor;
    float3 rayOrigin;
    float3 rayDir;
};

struct TraceResult {
    int missed = 0;

    int directLightId = -1;
    float distance;
    float3 position;
    float3 normal;
    float3 tangent;
    float2 texcoord;
    bool outer;

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
void RayTrace(float3 position, float3 rayDir, int rayType, TraceResult* result) {
    uint32_t u0, u1;
    PackPointer(result, u0, u1);
    optixTrace(
        renderParams.traversable,
        position,
        rayDir,
        0.001f, 1e20f, 0,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        rayType,
        RAY_TYPE_COUNT,
        rayType, u0, u1);
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
float3 RandomOnUnitSphere(unsigned int& seed) {
    while (true) {
        const float u = rnd(seed) * 2 - 1, v = rnd(seed) * 2 - 1;
        const float r2 = u * u + v * v;
        if (r2 >= 1.0f) continue;
        const float t = sqrt(1 - r2);
        return make_float3(2 * u * t, 2 * v * t, 1 - 2 * r2);
    }
}

static __forceinline__ __device__
float2 RandomInUnitDisk(unsigned int& seed) {
    while (true) {
        float2 p = make_float2(rnd(seed) * 2 - 1, rnd(seed) * 2 - 1);
        if (dot(p, p) >= 1) continue;
        return p;
    }
}

static __forceinline__ __device__
float3 RandomSampleOnHemisphere(unsigned int& seed, const float3& normal) {
    const float3 vecOnSphere = RandomOnUnitSphere(seed);
    if (dot(vecOnSphere, normal) > 0.0f)
        return vecOnSphere;
    else
        return -vecOnSphere;
}

static __forceinline__ __device__ float2 min(const float2 a, const float2 b) {
    return make_float2(min(a.x, b.x), min(a.y, b.y));
}

static __forceinline__ __device__ float2 max(const float2 a, const float2 b) {
    return make_float2(max(a.x, b.x), max(a.y, b.y));
}

static __forceinline__ __device__ float3 Refract(const float3& in, const float3& normal, float refractivity) {
    auto cos_theta = fminf(dot(-in, normal), 1.0f);
    float3 r_out_perp = refractivity * (in + cos_theta * normal);
    float3 r_out_parallel = -sqrtf(fabs(1.0f - dot(r_out_perp, r_out_perp))) * normal;
    return r_out_perp + r_out_parallel;
}

static __forceinline__ __device__ float CalcRefractivityForColor(float3 refractivity, float3 color) {
    return color.x > 0.5f ? refractivity.x : (color.y > 0.5f ? refractivity.y : refractivity.z);
}

static __forceinline__ __device__ float2 GetSphericalCoord(float3 vecOnSphere) {
    return make_float2(acosf(vecOnSphere.y), atan2f(vecOnSphere.z, vecOnSphere.x) + MPI);
}



static __forceinline__ __device__ float3 SampleLightInQuadWithoutBrdf(unsigned int& seed, 
                                const float3 p, const float3 normal, const float3 quadA, const float3 quadB, const float3 quadC, const float3 quadNormal, 
                                TraceResult& traceResult, const int targetDirectLightId, float3& sampleDir) {
    const float u = rnd(seed), v = rnd(seed);
    const float3 base1 = quadB - quadA, base2 = quadC - quadA;
    const float area = length(base1) * length(base2);

    float3 sample = quadA + u * base1 + v * base2;
    sampleDir = normalize(sample - p);

    if (dot(sampleDir, normal) < 0 || dot(quadNormal, -sampleDir) < 0) {
        return make_float3(0);
    }

    traceResult.directLightId = -1;
    RayTrace(p, sampleDir, RADIANCE_RAY_TYPE, &traceResult);

    if (traceResult.directLightId == targetDirectLightId && dot(traceResult.normal, quadNormal) > 0.99f && traceResult.missed == false) {
        return traceResult.material.emission * dot(normal, sampleDir) * dot(quadNormal, -sampleDir) / sqrtf(dot(sample - p, sample - p)) * area;
    }
    return make_float3(0);
}

#define SAMPLE_QUAD(v1, v2, v3, quadNormal, extraAttenuation) \
                { \
                ret = SampleLightInQuadWithoutBrdf(state.seed, traceResult.position, traceResult.normal, light.vertex[(v1)], light.vertex[(v2)], light.vertex[(v3)], (quadNormal), directLightTraceResult, i, sampleDir); \
                float3 delta = ret * extraAttenuation * state.attenuation; \
                result += dot(delta, delta) < renderParams.maxResultDeltaSqr ? delta : make_float3(0); \
                }

#define SAMPLE_DIRECT_LIGHT(result, state, traceResult, brdf) \
    {\
        SampleOnDirectLight(result, state, traceResult); \
        state.collectDirectLight = false; \
        \
        TraceResult directLightTraceResult;\
        \
        for (int i = 0; i < renderParams.directLightCount; i++) {\
            const DirectLightDescription& light = renderParams.deviceDirectLights[i]; \
            \
            float3 sampleDir; \
            float3 ret; \
            \
            SAMPLE_QUAD(0, 2, 4, make_float3(0, 0, -1), brdf); \
            SAMPLE_QUAD(0, 1, 2, make_float3(-1, 0, 0), brdf); \
            SAMPLE_QUAD(0, 1, 4, make_float3(0, -1, 0), brdf); \
            SAMPLE_QUAD(4, 5, 6, make_float3(1, 0, 0), brdf); \
            SAMPLE_QUAD(2, 3, 6, make_float3(0, 1, 0), brdf); \
            SAMPLE_QUAD(1, 3, 5, make_float3(0, 0, 1), brdf); \
            \
        }\
    }\

static __forceinline__ __device__ void SampleOnDirectLight(float3& result, PathState& state, TraceResult& traceResult) {
    
}