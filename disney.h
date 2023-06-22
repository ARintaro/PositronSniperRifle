#pragma once

#include "renderParams.h"
#include <crt\host_defines.h>
#include <optix_device.h>
#include "sutil\vec_math.h"


__constant__ float INV_PI = 1 / M_PI;
__constant__ float INV_2PI = 1 / (2 * M_PI);

static __forceinline__ __device__ float sqr(float x) {
    return x * x;
}

static __forceinline__ __device__ float SchlickFresnel(float u) {
    float m = clamp(1 - u, 0.f, 1.f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

static __forceinline__ __device__ float SchlickFresnelFeflectance(float cosine, float refractivity) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - refractivity) / (1 + refractivity);
    r0 = r0 * r0;
    return r0 + (1 - r0) * SchlickFresnel(cosine);
}

static __forceinline__ __device__  float GTR1(float NdotH, float a) {
    if (a >= 1) return INV_PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1.f) * NdotH * NdotH;
    return (a2 - 1.f) / (M_PI * logf(a2) * t);
}

static __forceinline__ __device__  float GTR2(float NdotH, float a) {
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (M_PI * t * t);
}

static __forceinline__ __device__ float lerp(const float a, const float b, const float t) {
    return a + (b - a) * t;
}

static __forceinline__ __device__ float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1.f / (M_PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

static __forceinline__ __device__ float SmithGGX(float NdotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrtf(a + b - a * b));
}

static __forceinline__ __device__ float SmithGggxAniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrtf(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}


static __forceinline__ __device__
float3 DisneyBRDF(const float3& V, const float3& N, const float3& L,
    const float3& X, const float3& Y,
                  const DisneyPbrData& data) {
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0) return make_float3(0);

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    float3 Cdlin = data.baseColor;
    float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : make_float3(1); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(data.specular * 0.08f * lerp(make_float3(1), Ctint, data.specularTint), Cdlin, data.metallic);
    float3 Csheen = lerp(make_float3(1), Ctint, data.sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * data.roughness;
    float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * data.roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float aspect = sqrt(1 - data.anisotropic * .9f);
    float ax = fmax(0.001f, data.roughness * data.roughness / aspect);
    float ay = fmax(0.001f, data.roughness * data.roughness * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, make_float3(1), FH);
    float Gs;
    Gs = SmithGggxAniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= SmithGggxAniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    float3 Fsheen = FH * data.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, lerp(.1f, .001f, data.clearcoatGloss));
    float Fr = lerp(0.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, 0.25f) * SmithGGX(NdotV, 0.25f);

    return (INV_PI * lerp(Fd, ss, data.subsurface) * Cdlin + Fsheen)
        * (1.f - data.metallic)
        + Gs * Fs * Ds + .25f * data.clearcoat * Gr * Fr * Dr;
}