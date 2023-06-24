#pragma once

#include "deviceHelper.cuh"


extern "C" __device__ void __direct_callable__naive_diffuse(float3& result, PathState& state, TraceResult& traceResult) {
    NaiveDiffuseData& data = *(NaiveDiffuseData*)traceResult.material.data;
    
    if (state.collectDirectLight || traceResult.directLightId == -1) {
        float3 delta = traceResult.material.emission * state.attenuation * state.supposedColor;
        if (dot(delta, delta) < 1e4f) {
            result += traceResult.material.emission * state.attenuation * state.supposedColor;
        }
    }

    if (traceResult.directLightId == -1) {
        SAMPLE_DIRECT_LIGHT(result, state, traceResult, data.albedo / MPI);
    }

    state.rayOrigin = traceResult.position;
    state.rayDir = RandomSampleOnHemisphere(state.seed, traceResult.normal);

    float cosine = dot(state.rayDir, traceResult.normal);

    state.attenuation *= 2 * cosine * data.albedo / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__naive_metal(float3 & result, PathState & state, TraceResult & traceResult) {
    NaiveMetalData& data = *(NaiveMetalData*)traceResult.material.data;

    state.rayOrigin = traceResult.position;
    state.rayDir = reflect(state.rayDir, traceResult.normal) + data.roughness * RandomInUnitSphere(state.seed);
    state.rayDir = normalize(state.rayDir);

    state.attenuation *= 1 / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__naive_dielectrics(float3 & result, PathState & state, TraceResult & traceResult) {
    NaiveDielectricsData& data = *(NaiveDielectricsData*)traceResult.material.data;

    state.rayOrigin = traceResult.position;

    float refractivity = traceResult.outer ? data.refractivity : 1.f / data.refractivity;
    float cosTheta = min(dot(-state.rayDir, traceResult.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    state.collectDirectLight = true;

    if (refractivity * sinTheta > 1.0f || SchlickFresnelFeflectance(cosTheta, refractivity) > rnd(state.seed)) {
        state.rayDir = reflect(state.rayDir, traceResult.normal);
    }
    else {
        state.rayDir = Refract(state.rayDir, traceResult.normal, refractivity);
    }

    state.attenuation *= 1 / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__disney_pbr(float3 & result, PathState & state, TraceResult & traceResult) {
    DisneyPbrData data = *(DisneyPbrData*)traceResult.material.data;

    if (data.baseColorTexture) {
        data.baseColor = make_float3(tex2D<float4>(data.baseColorTexture, traceResult.texcoord.x, traceResult.texcoord.y));
    }

    if (data.roughnessTexture) {
        data.roughness = tex2D<float4>(data.roughnessTexture, traceResult.texcoord.x, traceResult.texcoord.y).x;
    }

    if (data.metallicTexture) {
        data.metallic = tex2D<float4>(data.metallicTexture, traceResult.texcoord.x, traceResult.texcoord.y).x;
    }

    if (traceResult.directLightId == -1) {
        SAMPLE_DIRECT_LIGHT(result, state, traceResult, DisneyBRDF(-state.rayDir, traceResult.normal, sampleDir, data, traceResult));
    }

    const float3 beforeRaydir = state.rayDir;

    state.rayOrigin = traceResult.position;
    state.rayDir = RandomSampleOnHemisphere(state.seed, traceResult.normal);

    const float pdf = 1 / (2 * MPI);
    const float cosine = dot(state.rayDir, traceResult.normal);


    state.attenuation *= cosine * DisneyBRDF(state.rayDir, traceResult.normal, -beforeRaydir, data, traceResult) / pdf / renderParams.russianRouletteProbability;
}

extern "C" __device__ void __direct_callable__dispersion(float3 & result, PathState & state, TraceResult & traceResult) {
    // NaiveDielectricsData& data = *(NaiveDielectricsData*)traceResult.material.data;
    state.rayOrigin = traceResult.position;

    if (1.8f < state.supposedColor.x + state.supposedColor.y && state.supposedColor.x + state.supposedColor.y < 2.5f) {
        // ��������
        const float sample = rnd(state.seed) * 3;
        state.supposedColor = (sample > 2 ? make_float3(3, 0, 0) : (sample > 1 ? make_float3(0, 3, 0) : make_float3(0, 0, 3)));
    }
    else if (traceResult.outer) {
        traceResult.missed = true;
        return;
    }

    float refractivity = CalcRefractivityForColor(make_float3(1.2f, 1.3f, 1.5f), state.supposedColor);
    refractivity = traceResult.outer ? refractivity : 1.f / refractivity;

    float cosTheta = min(dot(-state.rayDir, traceResult.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    state.collectDirectLight = true;

    if (refractivity * sinTheta > 1.0f || SchlickFresnelFeflectance(cosTheta, refractivity) > rnd(state.seed)) {
        state.rayDir = reflect(state.rayDir, traceResult.normal);
    }
    else {
        state.rayDir = Refract(state.rayDir, traceResult.normal, refractivity);
    }

    state.attenuation *= 1 / renderParams.russianRouletteProbability;

}


