

#include "renderParams.h"


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


static __forceinline__ __device__ float CalculateX(const float v, const DeviceCurveData& data) {
    int n = data.n;
    float result = 0.f;
    for (int i = 0; i <= n; i++) {
        result += data.points[i].x * data.combs[i] * powf(v, i) * powf(1 - v, n - i);
    }
    // printf("%f %f\n", v, result);
    return result;
}

static __forceinline__ __device__ float CalculateY(const float v, const DeviceCurveData& data) {
    int n = data.n;
    float result = 0.f;
    for (int i = 0; i <= n; i++) {
        result += data.points[i].y * data.combs[i] * powf(v, i) * powf(1 - v, n - i);
    }
    return result;
}

static __forceinline__ __device__ float CalculateDx(const float v, const DeviceCurveData& data)
{
    int n = data.n;
    float result = 0.f;
    for (int i = 0; i <= n; i++)
    {
        result += data.points[i].x * data.combs[i] * (i * powf(v, i - 1) * powf(1 - v, n - i) + powf(v, i) * (i - n) * powf(1 - v, n - i - 1));
    }
    return result;
}

static __forceinline__ __device__ float CalculateDy(const float v, const DeviceCurveData& data)
{
    int n = data.n;
    float result = 0.f;
    for (int i = 0; i <= n; i++)
    {
        result += data.points[i].y * data.combs[i] * (i * powf(v, i - 1) * powf(1 - v, n - i) + powf(v, i) * (i - n) * powf(1 - v, n - i - 1));
    }
    return result;
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
    float3 normal = make_float3(0);

    if (data.normal) {
        float2 bary = optixGetTriangleBarycentrics();
        normal = (1.f - bary.x - bary.y) * data.normal[index.x] + bary.x * data.normal[index.y] + bary.y * data.normal[index.z];
    } else {
        const float3& A = data.vertex[index.x];
        const float3& B = data.vertex[index.y];
        const float3& C = data.vertex[index.z];
        normal = normalize(cross(B - A, C - A));
    }

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

    const float3 rayDir = optixGetWorldRayDirection();

    result.missed = false;
    result.normal = normal;
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

extern "C" __global__ void __closesthit__curve() {
    TraceResult& result = *GetPerRayData<TraceResult>();

    const ShaderBindingData& sbtData = *(const ShaderBindingData*)optixGetSbtDataPointer();
    const Material& material = sbtData.material;
    const DeviceCurveData& data = sbtData.data.curve;

    const float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 normal = make_float3(
        __int_as_float(optixGetAttribute_0()),
        __int_as_float(optixGetAttribute_1()),
        __int_as_float(optixGetAttribute_2())
    );

    

    result.missed = false;
    result.normal = normal;
    result.position = position;
    result.material = material;
}

extern "C" __global__ void __intersection__curve() {
    const ShaderBindingData& sbtData = *(const ShaderBindingData*)optixGetSbtDataPointer();
    const DeviceCurveData& data = sbtData.data.curve;
    const float3 origin = optixGetObjectRayOrigin();
    const float3 direction = optixGetObjectRayDirection();

    float t = 1e10;
    float3 tmax = { data.aabb->maxX, data.aabb->maxY, data.aabb->maxZ };
    float3 tmin = { data.aabb->minX, data.aabb->minY, data.aabb->minZ };

    tmax = (tmax - origin) / direction;
    tmin = (tmin - origin) / direction;

    if ((origin + tmax.x * direction).y > data.aabb->minY && (origin + tmax.x * direction).y < data.aabb->maxY && (origin + tmax.x * direction).z > data.aabb->minZ && (origin + tmax.x * direction).z < data.aabb->maxZ && tmax.x < t)
    {
        t = tmax.x;
    }

    if ((origin + tmax.y * direction).x > data.aabb->minX && (origin + tmax.y * direction).x < data.aabb->maxX && (origin + tmax.y * direction).z > data.aabb->minZ && (origin + tmax.y * direction).z < data.aabb->maxZ && tmax.y < t)
    {
        t = tmax.y;
    }

    if ((origin + tmax.z * direction).y > data.aabb->minX && (origin + tmax.z * direction).x < data.aabb->maxX && (origin + tmax.z * direction).y > data.aabb->minY && (origin + tmax.z * direction).y < data.aabb->maxY && tmax.z < t)
    {
        t = tmax.z;
    }

    if ((origin + tmin.x * direction).y > data.aabb->minY && (origin + tmin.x * direction).y < data.aabb->maxY && (origin + tmin.x * direction).z > data.aabb->minZ && (origin + tmin.x * direction).z < data.aabb->maxZ && tmin.x < t)
    {
        t = tmin.x;
    }

    if ((origin + tmin.y * direction).x > data.aabb->minX && (origin + tmin.y * direction).x < data.aabb->maxX && (origin + tmin.z * direction).z > data.aabb->minZ && (origin + tmin.x * direction).z < data.aabb->maxZ && tmin.y < t)
    {
        t = tmin.y;
    }

    if ((origin + tmin.z * direction).z > data.aabb->minX && (origin + tmin.z * direction).x < data.aabb->maxX && (origin + tmin.z * direction).y > data.aabb->minY && (origin + tmin.z * direction).y < data.aabb->maxY && tmin.z < t)
    {
        t = tmin.z;
    }

    float3 bvh_intersect = origin + t * direction;
    bvh_intersect = bvh_intersect - data.position;
    float3 init_xyz = {};
    init_xyz.x = cos(data.theta) * bvh_intersect.x + sin(data.theta) * bvh_intersect.y;
    init_xyz.y = -sin(data.theta) * bvh_intersect.x + cos(data.theta) * bvh_intersect.y;
    init_xyz.z = bvh_intersect.z;

    float3 init_uvt = {};
    init_uvt.x = atan(init_xyz.z / init_xyz.x);
    if (init_xyz.x > 0 && init_xyz.z < 0)
    {
        init_uvt.x += 2 * M_PI;
    }
    else if (init_xyz.x < 0)
    {
        init_uvt.x += M_PI;
    }
    init_uvt.y = (init_xyz.y - data.aabb->minY) / (data.aabb->maxY - data.aabb->minY);
    if (init_uvt.y < 0)
    {
        init_uvt.y = 0.00001f;
    }
    else if (init_uvt.y > 1)
    {
        init_uvt.y = 0.99999f;
    }
    init_uvt.z = t;

    float3 intersect_uvt = init_uvt;

    int max_round = 5;
    float delta = 0.1;
    float epsilon = 0.00001;

    for (int i = 0; i < max_round; i++)
    {
        float3 Jocab_first_row = { -CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x), CalculateDx(intersect_uvt.y, data) * cos(intersect_uvt.x), -direction.x };
        float3 Jocab_second_row = { 0, CalculateDy(intersect_uvt.y, data), -direction.y };
        float3 Jocab_third_row = { CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x), CalculateDx(intersect_uvt.y, data) * sin(intersect_uvt.x), -direction.z };

        float3 F_uvt = { CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x), CalculateY(intersect_uvt.y, data), CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x) };
        F_uvt = F_uvt - origin - direction * intersect_uvt.z;

        float3 d_uvt = {};
        if (abs(Jocab_first_row.x) > epsilon)
        {
            float coefficient = (Jocab_third_row.x / Jocab_first_row.x);
            Jocab_third_row -= coefficient * Jocab_first_row;
            F_uvt.z -= coefficient * F_uvt.x;
        }

        if (abs(Jocab_second_row.y) > epsilon)
        {
            float coefficient = (Jocab_third_row.y / Jocab_second_row.y);
            Jocab_third_row -= coefficient * Jocab_second_row;
            F_uvt.z -= coefficient * F_uvt.y;
        }

        if (abs(Jocab_third_row.z) > epsilon)
        {
            d_uvt.z = F_uvt.z / Jocab_third_row.z;
        }
        else
        {
            return;
        }

        d_uvt.y = (F_uvt.y - Jocab_second_row.z * d_uvt.z) / Jocab_second_row.y;
        d_uvt.x = (F_uvt.x - Jocab_first_row.z * d_uvt.z - Jocab_first_row.y * d_uvt.y) / Jocab_first_row.x;
        intersect_uvt -= d_uvt;

        if (intersect_uvt.y < 0)
        {
            intersect_uvt.y = 0.00001;
        }
        else if (intersect_uvt.y > 1)
        {
            intersect_uvt.y = 0.99999;
        }
        if (intersect_uvt.x < 0)
        {
            intersect_uvt.x = 0.f;
        }
        else if (intersect_uvt.x > 2 * M_PI)
        {
            intersect_uvt.x = 2 * M_PI;
        }
        if (dot(d_uvt, d_uvt) < delta)
        {
            break;
        }
    }

    float3 intersect_xyz = {};
    intersect_xyz.x = CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x);
    intersect_xyz.y = CalculateY(intersect_uvt.y, data);
    intersect_xyz.z = CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x);

    intersect_xyz.x = cos(data.theta) * intersect_xyz.x - sin(data.theta) * intersect_xyz.y;
    intersect_xyz.y = sin(data.theta) * intersect_xyz.x + cos(data.theta) * intersect_xyz.y;
    intersect_xyz += data.position;

    t = intersect_uvt.z;
    float res = 0.01f;
    if (length(origin + t * direction - intersect_xyz) > res)
    {
        return;
    }
    float3 tangent = { CalculateDx(intersect_uvt.y, data), CalculateDy(intersect_uvt.y, data), 0 };
    tangent = normalize(tangent);
    float3 norm = { -tangent.y, tangent.x, 0 };
    norm = { norm.x * cos(intersect_uvt.x), norm.y, norm.x * sin(intersect_uvt.x) };
    norm = { norm.x * cos(data.theta) - norm.y * sin(data.theta),
            norm.x * sin(data.theta) + norm.y * cos(data.theta),
            norm.z };
    if (dot(norm, direction) > 0)
    {
        norm = -norm;
    }

    optixReportIntersection(t, 0, __float_as_int(norm.x), __float_as_int(norm.y), __float_as_int(norm.z));
}




extern "C" __device__ void __direct_callable__naive_diffuse(unsigned int& seed, float3& result, TraceResult& traceResult, float3& attenuation, float3& rayOrigin, float3& rayDir) {
    NaiveDiffuseData& data = *(NaiveDiffuseData*)traceResult.material.data;

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