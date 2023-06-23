

#include "renderParams.h"
#include "random.h"
#include "disney.h"

extern "C" __constant__ RenderParams renderParams;

struct TraceResult {
    int missed = 0;

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
float3 RandomInUnitSphere(unsigned int& seed) {
    while (true) {
        float3 v = make_float3(rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f, rnd(seed) * 2.0f - 1.0f);
        if (dot(v, v) >= 1.0f) continue;
        return v;
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
float3 RandomSampleHemisphere(unsigned int& seed, const float3& normal) {
    const float3 vec_in_sphere = RandomInUnitSphere(seed);
    if (dot(vec_in_sphere, normal) > 0.0f)
        return vec_in_sphere;
    else
        return -vec_in_sphere;
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


static __forceinline__ __device__ float3 Newton(float t, const float3& origin, const float3& direction, const DeviceCurveData& data)
{
    float3 init_xyz = origin + t * direction - data.position;

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

    int max_round = 10;
    float epsilon = 0.001;

    for (int i = 0; i < max_round; i++)
    {
        float3 Jocab_first_row = { -CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x), CalculateDx(intersect_uvt.y, data) * cos(intersect_uvt.x), -direction.x };
        float3 Jocab_second_row = { 0, CalculateDy(intersect_uvt.y, data), -direction.y };
        float3 Jocab_third_row = { CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x), CalculateDx(intersect_uvt.y, data) * sin(intersect_uvt.x), -direction.z };

        float3 F_uvt = { CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x), CalculateY(intersect_uvt.y, data), CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x) };
        F_uvt = F_uvt - origin - direction * intersect_uvt.z;

        float3 d_uvt = {};
        if (abs(Jocab_first_row.x) > epsilon) {
            float coefficient = (Jocab_third_row.x / Jocab_first_row.x);
            Jocab_third_row -= coefficient * Jocab_first_row;
            F_uvt.z -= coefficient * F_uvt.x;
        } else {
            return;
        }

        if (abs(Jocab_second_row.y) > epsilon) {
            float coefficient = (Jocab_third_row.y / Jocab_second_row.y);
            Jocab_third_row -= coefficient * Jocab_second_row;
            F_uvt.z -= coefficient * F_uvt.y;
        } else {
            return;
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

        if (intersect_uvt.y < 0) {
            intersect_uvt.y = 0.00001;
        } else if (intersect_uvt.y > 1) {
            intersect_uvt.y = 0.99999;
        } 
    }
    return intersect_uvt;
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

        float2 unitDisk = camera.lenRadius * RandomInUnitDisk(seed);

        float3 rayOrigin = camera.position + unitDisk.x * normalize(camera.horizontal) + unitDisk.y * normalize(camera.vertical);
        float3 rayDir = normalize(camera.direction + (screenPos.y - 0.5f) * camera.vertical + (screenPos.x - 0.5f) * camera.horizontal);

        float3 attenuation = make_float3(1.f);
        float3 supposedColor = make_float3(1.f);
       

        for (int depth = 0; depth < renderParams.maxDepth; depth++) {
            traceResult.missed = true;

            if (rnd(seed) > renderParams.russianRouletteProbability) {
                break;
            }

            RayTrace(rayOrigin, rayDir, RADIANCE_RAY_TYPE, &traceResult);

            if (traceResult.missed) {
                break;
            }

            optixDirectCall<void, unsigned int&, float3&, TraceResult&, float3&, float3&, float3&, float3&>(traceResult.material.programIndex, seed, result,traceResult, attenuation, rayOrigin, rayDir, supposedColor);
            
            if (traceResult.missed) {
                break;
            }
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

    if (dot(rayDir, normal) > 0) {
        normal = -normal;
        result.outer = false;
    }

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
    float3 origin = optixGetObjectRayOrigin();
    float3 direction = optixGetObjectRayDirection();

    float3 tmax = { data.aabb->maxX, data.aabb->maxY, data.aabb->maxZ };
    float3 tmin = { data.aabb->minX, data.aabb->minY, data.aabb->minZ };

    tmax = (tmax - origin) / direction;
    tmin = (tmin - origin) / direction;

    float3 intersect_uvt = { 1e10, 1e10, 1e10 };
    origin -= data.position;

    float3 from_tmax_x = Newton(tmax.x, origin, direction, data);
    float3 from_tmax_y = Newton(tmax.y, origin, direction, data);
    float3 from_tmax_z = Newton(tmax.z, origin, direction, data);
    float3 from_tmin_x = Newton(tmin.x, origin, direction, data);
    float3 from_tmin_y = Newton(tmin.y, origin, direction, data);
    float3 from_tmin_z = Newton(tmin.z, origin, direction, data);

    if (from_tmax_x.z > 0 && from_tmax_x.z < intersect_uvt.z) {
        intersect_uvt = from_tmax_x;
    }
    if (from_tmax_y.z > 0 && from_tmax_y.z < intersect_uvt.z) {
        intersect_uvt = from_tmax_y;
    }
    if (from_tmax_z.z > 0 && from_tmax_z.z < intersect_uvt.z) {
        intersect_uvt = from_tmax_z;
    }
    if (from_tmin_x.z > 0 && from_tmin_x.z < intersect_uvt.z) {
        intersect_uvt = from_tmin_x;
    }
    if (from_tmin_y.z > 0 && from_tmin_y.z < intersect_uvt.z) {
        intersect_uvt = from_tmin_y;
    }
    if (from_tmin_z.z > 0 && from_tmin_z.z < intersect_uvt.z) {
        intersect_uvt = from_tmin_z;
    }

    float3 intersect_xyz = {};
    intersect_xyz.x = CalculateX(intersect_uvt.y, data) * cos(intersect_uvt.x);
    intersect_xyz.y = CalculateY(intersect_uvt.y, data);
    intersect_xyz.z = CalculateX(intersect_uvt.y, data) * sin(intersect_uvt.x);

    float res = 0.01f;
    if (length(origin + intersect_uvt.z * direction - intersect_xyz) > res)
    {
        return;
    }
    intersect_xyz += data.position;
    float3 tangent = { CalculateDx(intersect_uvt.y, data), CalculateDy(intersect_uvt.y, data), 0 };
    tangent = normalize(tangent);
    float3 norm = { -tangent.y, tangent.x, 0 };
    norm = { norm.x * cos(intersect_uvt.x), norm.y, norm.x * sin(intersect_uvt.x) };

    if (dot(norm, direction) > 0)
    {
        norm = -norm;
    }

    optixReportIntersection(intersect_uvt.z, 0, __float_as_int(norm.x), __float_as_int(norm.y), __float_as_int(norm.z));
}

extern "C" __device__ void __direct_callable__naive_diffuse(unsigned int& seed, float3& result, TraceResult& traceResult, float3& attenuation, float3& rayOrigin, float3& rayDir, float3& supposedColor) {
    NaiveDiffuseData& data = *(NaiveDiffuseData*)traceResult.material.data;

    result += data.emission * attenuation * supposedColor;

    rayOrigin = traceResult.position;
    rayDir = RandomSampleHemisphere(seed, traceResult.normal);

    float cosine = dot(rayDir, traceResult.normal);

    // attenuation *= 2 * cosine * data.albedo / renderParams.russianRouletteProbability;
    attenuation *= data.albedo / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__naive_metal(unsigned int& seed, float3 & result, TraceResult & traceResult, float3 & attenuation, float3 & rayOrigin, float3 & rayDir, float3& supposedColor) {
    NaiveMetalData& data = *(NaiveMetalData*)traceResult.material.data;

    rayOrigin = traceResult.position;
    rayDir = reflect(rayDir, traceResult.normal) + data.roughness * RandomInUnitSphere(seed);
    rayDir = normalize(rayDir);

    attenuation *= 1 / renderParams.russianRouletteProbability;
}


extern "C" __device__ void __direct_callable__naive_dielectrics(unsigned int& seed, float3 & result, TraceResult & traceResult, float3 & attenuation, float3 & rayOrigin, float3 & rayDir, float3& supposedColor) {
    NaiveDielectricsData& data = *(NaiveDielectricsData*)traceResult.material.data;

    rayOrigin = traceResult.position;

    float refractivity = traceResult.outer ? data.refractivity : 1.f / data.refractivity;
    float cosTheta = min(dot(-rayDir, traceResult.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    if (refractivity * sinTheta > 1.0f || SchlickFresnelFeflectance(cosTheta, refractivity) > rnd(seed)) {
        rayDir = reflect(rayDir, traceResult.normal);
    } else {
        rayDir = Refract(rayDir, traceResult.normal, refractivity);
    }

    attenuation *= 1 / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__disney_pbr(unsigned int& seed, float3 & result, TraceResult & traceResult, float3 & attenuation, float3 & rayOrigin, float3 & rayDir, float3& supposedColor) {
    DisneyPbrData& data = *(DisneyPbrData*)traceResult.material.data;

    const float3 beforeRayDir = rayDir;

    rayOrigin = traceResult.position;
    rayDir = RandomSampleHemisphere(seed, traceResult.normal);

    const float pdf = 1 / (2 * M_PI);
    const float cosine = dot(rayDir, traceResult.normal);

    // Make a fake tanget
    float3 tangentHelper = abs(traceResult.normal.x > 0.99) ? make_float3(0, 0, 1) : make_float3(1, 0, 0);

    float3 bitangent = normalize(cross(traceResult.normal, tangentHelper));
    float3 tangent = normalize(cross(traceResult.normal, bitangent));
   
    attenuation *= cosine * DisneyBRDF(rayDir, traceResult.normal, -beforeRayDir, tangent, bitangent, data) / pdf / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__dispersion(unsigned int& seed, float3 & result, TraceResult & traceResult, float3 & attenuation, float3 & rayOrigin, float3 &rayDir, float3& supposedColor) {
    // NaiveDielectricsData& data = *(NaiveDielectricsData*)traceResult.material.data;
    rayOrigin = traceResult.position;

    if (1.8f < supposedColor.x + supposedColor.y && supposedColor.x + supposedColor.y < 2.5f) {
        // ³õ´ÎÕÛÉä
        const float sample = rnd(seed) * 3;
        supposedColor = (sample > 2 ? make_float3(3, 0, 0) : (sample > 1 ? make_float3(0, 3, 0) : make_float3(0, 0, 3)));
    }
    else if (traceResult.outer) {
        traceResult.missed = true;
        return;
    }
    
    float refractivity = CalcRefractivityForColor(make_float3(1.2f, 1.3f, 1.5f), supposedColor);
    refractivity = traceResult.outer ? refractivity : 1.f / refractivity;

    float cosTheta = min(dot(-rayDir, traceResult.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    if (refractivity * sinTheta > 1.0f || SchlickFresnelFeflectance(cosTheta, refractivity) > rnd(seed)) {
        rayDir = reflect(rayDir, traceResult.normal);
    }
    else {
        rayDir = Refract(rayDir, traceResult.normal, refractivity);
    }

    attenuation *= 1 / renderParams.russianRouletteProbability;

}


