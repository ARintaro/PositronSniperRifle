#pragma once

#include "deviceHelper.cuh"


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
        init_uvt.x += 2 * MPI;
    }
    else if (init_xyz.x < 0)
    {
        init_uvt.x += MPI;
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
        }
        else {
            return;
        }

        if (abs(Jocab_second_row.y) > epsilon) {
            float coefficient = (Jocab_third_row.y / Jocab_second_row.y);
            Jocab_third_row -= coefficient * Jocab_second_row;
            F_uvt.z -= coefficient * F_uvt.y;
        }
        else {
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
        }
        else if (intersect_uvt.y > 1) {
            intersect_uvt.y = 0.99999;
        }
    }
    return intersect_uvt;
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
    result.distance = optixGetRayTmax();
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

