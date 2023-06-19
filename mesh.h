#pragma once

#include <vector>
#include "renderParams.h"

class Mesh {

public:
	void AddCube(const float3& center, const float3& halfSize);

	void AddTriangle(const float3& a, const float3& b, const float3& c);

	void Rotate(float angle, const float3& axis);

	float3 color;
	std::vector<float3> vertex;
	std::vector<int3> index;

	Material material;
};