#pragma once

#include "renderParams.h"
#include <crt\host_defines.h>
#include <optix_device.h>
#include "sutil\vec_math.h"
#include <random.h>

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

    int directLightId = 0;
    float distance;
    float3 position;
    float3 normal;
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
        0.001, 1e20f, 0,
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

// (theta, phi)
static __forceinline__ __device__ float2 GetSphericalCoord(float3 vecOnSphere) {
    return make_float2(acosf(vecOnSphere.y), atanf(vecOnSphere.z / vecOnSphere.x));
}


// (theta_min, phi_min, theta_max, phi_max)
static __forceinline__ __device__ void GetScRange(const DirectLightDescription& light, const float3& position, const float3& normal, float2& scMin, float2& scMax, bool& existed) {
    existed = false;
    for (int i = 0; i < 8; i++) {
        float3 dir = normalize(light.vertex[i] - position);
        if (dot(dir, normal) < 0) continue;

        float2 sc = GetSphericalCoord(dir);

        if (existed == false) {
            scMin = scMax = sc;
        } else {
            scMin = min(scMin, sc);
            scMax = max(scMax, sc);
        }

        existed = true;
    }
}

// (theta, phi, pdf)
static __forceinline__ __device__ float3 SampleOnScRange(unsigned int& seed, float2 scMin, float2 scMax) {
    float theta = (scMax.x - scMin.x) * rnd(seed) + scMin.x;
    float phi = (scMax.y - scMin.y) * rnd(seed) + scMin.y;
    float pdf = (scMax.x - scMin.x) * (scMax.y - scMin.y);
    
    return make_float3(theta, phi, pdf);
}

static __forceinline__ __device__ void SampleOnDirectLight(float3& result, PathState& state, TraceResult& traceResult) {
    state.collectDirectLight = false;

    // TODO
    const float brdf = 1.f;

    TraceResult directLightTraceResult;
    directLightTraceResult.directLightId = -1;

    for (int i = 0; i < renderParams.directLightCount; i++) {
        const DirectLightDescription& light = renderParams.deviceDirectLights[i];
        
        bool existed;
        float2 scMin, scMax;
        GetScRange(light, traceResult.position, traceResult.normal, scMin, scMax, existed);

        if (existed == false) continue;

        // TODO ¾ùÔÈ²ÉÑù
        float3 sample = SampleOnScRange(state.seed, scMin, scMax);

        float sinTheta = sinf(sample.x);

        float3 dir = make_float3(sinTheta * cosf(sample.y), cosf(sample.x), sinTheta * sinf(sample.y));

        RayTrace(traceResult.position, dir, RADIANCE_RAY_TYPE, &directLightTraceResult);

        if (directLightTraceResult.directLightId == i) {
            result += directLightTraceResult.material.emission * state.attenuation * state.supposedColor * dot(directLightTraceResult.normal, -dir) * brdf;;
        }
    }
}