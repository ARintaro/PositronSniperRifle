#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>


enum { RADIANCE_RAY_TYPE = 0, /* OCCLUSION_RAY_TYPE = 1,*/ RAY_TYPE_COUNT };

enum { LAMBERT_MATERIAL_TYPE = 0, MIRROR_MATERIAL_TYPE = 1};

enum { MESH_OBJECT_TYPE = 0, SPHERE_OBJECT_TYPE = 1, CURVE_OBJECT_TYPE = 2 , OBJECT_TYPE_COUNT };

using Color = uchar4;

struct DisneyPbrData {
	float3 baseColor = make_float3(1, 1, 1);
	float roughness = 1;
	float metallic = 0; 
	float specular = 0;
	float specularTint = 0;

	float sheenTint = 0;
	float anisotropic = 0;
	float sheen = 0;
	float clearcoatGloss = 0;
	float subsurface = 0;
	float clearcoat = 0;
};

struct NaiveDiffuseData {
	float3 emission = make_float3(0, 0, 0);
	float3 albedo = make_float3(1, 1, 1);
};

struct NaiveMetalData {
	float roughness = 0;
};

struct NaiveDielectricsData {
	float refractivity;
};

struct Material {
	void* data;
	int programIndex;
};

struct DeviceMeshData {
	float3* vertex;
	float3* normal;
	int3* index;
};

struct DeviceSphereData {
	float3 position;
	float radius;
};

struct DeviceCurveData {
	int n;
	float3* points;
	int* combs;
	float3 position;
	float theta;
	OptixAabb* aabb;
};

struct ShaderBindingData {
	union Data {
		DeviceMeshData mesh;
		DeviceSphereData sphere;
		DeviceCurveData curve;
	}data;

	Material material;
};

struct RenderParams {
	int2 screenSize {512, 512};

	int samplesPerLaunch = 32;

	float russianRouletteProbability = 0.9f;

	float3 skyLightDirection = make_float3(0, -1, 1);
	float3 skyLightColor = make_float3(1.f, 1.f, 1.f);

	int maxDepth = 16;

	struct Frame {
		int frameId = 0;
		int subframeCount = 0;
		float3* colorBuffer = nullptr;
	}frame;

	struct Camera {
		float3 position;
		float3 direction;
		float3 horizontal;
		float3 vertical;

		float lenRadius;
	}camera;

	OptixTraversableHandle traversable;

};