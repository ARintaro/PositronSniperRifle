#pragma once

#include "optixLib.h"

enum { RADIANCE_RAY_TYPE = 0, /* OCCLUSION_RAY_TYPE = 1,*/ RAY_TYPE_COUNT };

enum { LAMBERT_MATERIAL_TYPE = 0, MIRROR_MATERIAL_TYPE = 1};

enum { MESH_OBJECT_TYPE = 0, SPHERE_OBJECT_TYPE = 1, OBJECT_TYPE_COUNT };

using Color = uchar4;

struct NaviveDiffuseData {
	float3 emission = make_float3(0, 0, 0);
	float3 albedo = make_float3(1, 1, 1);
};

struct NaiveMirrorData {

};

struct Material {
	void* data;
	int programIndex;
};

struct DeviceMeshData {
	float3* vertex;
	int3* index;
};

struct DeviceSphereData {
	float3 position;
	float radius;
};

struct ShaderBindingData {
	union Data {
		DeviceMeshData mesh;
		DeviceSphereData sphere;
	}data;

	Material material;
};

struct RenderParams {
	int2 screenSize {512, 512};

	int samplesPerLaunch = 128;

	float russianRouletteProbability = 0.9f;

	int maxDepth = 32;

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