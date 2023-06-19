#pragma once

#include "optixLib.h"

enum { RADIANCE_RAY_TYPE = 0, /* OCCLUSION_RAY_TYPE = 1,*/ RAY_TYPE_COUNT };

enum { LAMBERT_MATERIAL_TYPE = 0, MIRROR_MATERIAL_TYPE = 1};

using Color = uchar4;

struct Material {
	float3 albedo = make_float3(0, 0, 0);
	float3 emission = make_float3(0, 0, 0);

	int programIndex;
};

struct MeshShaderBindingData {
	float3* vertex;
	int3* index;

	Material material;
};

struct RenderParams {
	int2 screenSize {512, 512};

	int samplesPerLaunch = 16;

	float russianRouletteProbability = 0.9f;

	int maxDepth = 512;

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
	}camera;

	OptixTraversableHandle traversable;

};