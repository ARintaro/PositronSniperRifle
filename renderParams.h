#pragma once

#include "optixLib.h"

enum { RADIANCE_RAY_TYPE = 0, /* OCCLUSION_RAY_TYPE = 1,*/ RAY_TYPE_COUNT };

enum { LAMBERT_MATERIAL_TYPE = 0, MIRROR_MATERIAL_TYPE = 1};

using Color = uchar4;

struct Material {
	int programIndex;
};

struct MeshShaderBindingData {
	float3 color;
	float3* vertex;
	int3* index;

	Material material;
};

struct RenderParams {
	int2 screenSize {512, 512};

	int samplesPerLaunch = 4;

	int maxDepth = 4;

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