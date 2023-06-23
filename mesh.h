#pragma once

#include <vector>
#include "renderParams.h"
#include <positronSniperRifle\cudaBuffer.h>
#include "shader.hpp"
#include <map>
#include <set>
#include "tiny_obj_loader.h"


struct HitgroupRecord;

class SceneObject {

public:
	std::shared_ptr<HostMaterial> material;
	uint32_t geometryFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) = 0;

	virtual void GetBuildInput(OptixBuildInput& input) = 0;

};

class Mesh : public SceneObject {

public:
	static std::vector<Mesh> LoadObj(const std::string& fileName);
	
	void AddCube(const float3& center, const float3& halfSize);

	void AddTriangle(const float3& a, const float3& b, const float3& c);

	void AddPrism(const float3& center, const float halfsize, const float3& vertix);

	void Rotate(float angle, const float3& axis);

	void Move(const float3& delta);

	void Scale(const float3& scale);

	virtual void GetBuildInput(OptixBuildInput& input) override;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) override;

	int AddVertex(tinyobj::attrib_t& attributes, const tinyobj::index_t& idx, std::map<tinyobj::index_t, int>& knownVertices);


protected:
	std::vector<float3> vertex;
	std::vector<float3> normal;
	std::vector<float2> texcoord;
	std::vector<int3> index;

	CudaBuffer deviceVertex;
	CudaBuffer deviceIndex;
	CudaBuffer deviceNormal;
};

class Sphere : public SceneObject {

public:
	float3 position;
	float radius;

	virtual void GetBuildInput(OptixBuildInput& input) override;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) override;

protected:
	CudaBuffer devicePosition;
	CudaBuffer deviceRadius;

};

class Curve : public SceneObject {
public:
	std::vector<float3> points;
	float theta;
	float3 position;

	virtual void GetBuildInput(OptixBuildInput& input) override;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) override;

	OptixAabb GetAabb();

protected:
	OptixAabb aabb;

	std::vector<int> combs;

	CudaBuffer devicePoints;
	CudaBuffer deviceAabb;
	CudaBuffer deviceCombs;

	void CalculateComb();

	float CalculateX(float v);

	float CalculateY(float v);

	void ZRotate(float3& vec);

	void GetRange(float3& min_range, float3& max_range);
};

