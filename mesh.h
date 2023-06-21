#pragma once

#include <vector>
#include "renderParams.h"
#include <positronSniperRifle\cudaBuffer.h>
#include "shader.hpp"


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
	void AddCube(const float3& center, const float3& halfSize);

	void AddTriangle(const float3& a, const float3& b, const float3& c);

	void Rotate(float angle, const float3& axis);

	void Move(const float3& delta);

	virtual void GetBuildInput(OptixBuildInput& input) override;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) override;

	
protected:
	std::vector<float3> vertex;
	std::vector<int3> index;

	CudaBuffer deviceVertex;
	CudaBuffer deviceIndex;
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
	float3 axis;
	float3 position;
	OptixAabb aabb;

	virtual void GetBuildInput(OptixBuildInput& input) override;

	virtual void GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) override;

	OptixAabb GetAabb();

protected:
	CudaBuffer devicePoints;
	CudaBuffer deviceAabb;
};

