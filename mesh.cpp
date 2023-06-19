#include "mesh.h"
#include "sutil/vec_math.h"
#include "sutil/Matrix.h"

void Mesh::AddCube(const 
	float3 & center, const float3& halfSize) {
	int3 firstVertexId = make_int3((int)vertex.size());

	vertex.push_back(center + make_float3(-1, -1, -1) * halfSize); // 0
	vertex.push_back(center + make_float3(-1, 1, -1) * halfSize); // 1
	vertex.push_back(center + make_float3(-1, 1, 1) * halfSize); // 2
	vertex.push_back(center + make_float3(-1, -1, 1) * halfSize); // 3
	vertex.push_back(center + make_float3(1, 1, 1) * halfSize); // 4
	vertex.push_back(center + make_float3(1, -1, 1) * halfSize); // 5
	vertex.push_back(center + make_float3(1, 1, -1) * halfSize); // 6
	vertex.push_back(center + make_float3(1, -1, -1) * halfSize); // 7

	index.push_back(firstVertexId + make_int3(0, 1, 2));
	index.push_back(firstVertexId + make_int3(0, 2, 3));

	index.push_back(firstVertexId + make_int3(2, 5, 3));
	index.push_back(firstVertexId + make_int3(2, 4, 5));

	index.push_back(firstVertexId + make_int3(4, 6, 5));
	index.push_back(firstVertexId + make_int3(7, 5, 6));

	index.push_back(firstVertexId + make_int3(0, 6, 1));
	index.push_back(firstVertexId + make_int3(0, 7, 6));

	index.push_back(firstVertexId + make_int3(1, 6, 2));
	index.push_back(firstVertexId + make_int3(2, 6, 4));

	index.push_back(firstVertexId + make_int3(0, 3, 7));
	index.push_back(firstVertexId + make_int3(3, 5, 7));
}

void Mesh::AddTriangle(const float3& a, const float3& b, const float3& c) {
	int3 firstVertexId = make_int3((int)vertex.size());

	vertex.push_back(a);
	vertex.push_back(b);
	vertex.push_back(c);

	index.push_back(firstVertexId + make_int3(0, 1, 2));
}

void Mesh::Rotate(float angle, const float3& axis) {
	auto matrix = sutil::Matrix<4, 4>::rotate(angle * M_PI / 180, axis);

	for (auto& v : vertex) {
		auto v4 = make_float4(v, 1.f) * matrix;
		v = make_float3(v4);
	}
}

