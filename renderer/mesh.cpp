#define TINYOBJLOADER_IMPLEMENTATION

#include "mesh.h"
#include "sutil/vec_math.h"
#include "sutil/Matrix.h"
#include "renderer.h"



namespace std {
	inline bool operator < (const tinyobj::index_t& a,
		const tinyobj::index_t& b)
	{
		if (a.vertex_index < b.vertex_index) return true;
		if (a.vertex_index > b.vertex_index) return false;

		if (a.normal_index < b.normal_index) return true;
		if (a.normal_index > b.normal_index) return false;

		if (a.texcoord_index < b.texcoord_index) return true;
		if (a.texcoord_index > b.texcoord_index) return false;

		return false;
	}
}

int Mesh::AddVertex(tinyobj::attrib_t& attributes, const tinyobj::index_t& idx, std::map<tinyobj::index_t, int>& knownVertices) {
	if (knownVertices.find(idx) != knownVertices.end())
		return knownVertices[idx];
	const float3* vertex_array = (const float3*)attributes.vertices.data();
	const float3* normal_array = (const float3*)attributes.normals.data();
	const float2* texcoord_array = (const float2*)attributes.texcoords.data();

	int newID = vertex.size();
	knownVertices[idx] = newID;
	vertex.push_back(vertex_array[idx.vertex_index]);

	if (idx.normal_index >= 0) {
		while (normal.size() < vertex.size())
			normal.push_back(normal_array[idx.normal_index]);
	}
	if (idx.texcoord_index >= 0) {
		while (texcoord.size() < vertex.size())
			texcoord.push_back(texcoord_array[idx.texcoord_index]);
	}
	// just for sanity's sake:
	if (texcoord.size() > 0)
		texcoord.resize(vertex.size());
	// just for sanity's sake:
	if (normal.size() > 0)
		normal.resize(vertex.size());

	return newID;
}

OptixAabb Mesh::GetAabb() {
	OptixAabb aabb;
	aabb.minX = aabb.maxX = vertex[0].x;
	aabb.minY = aabb.maxY = vertex[0].y;
	aabb.minZ = aabb.maxZ = vertex[0].z;

	for (auto& v : vertex) {
		aabb.minX = std::min(aabb.minX, v.x);
		aabb.maxX = std::max(aabb.maxX, v.x);

		aabb.minY = std::min(aabb.minY, v.y);
		aabb.maxY = std::max(aabb.maxY, v.y);

		aabb.minZ = std::min(aabb.minZ, v.z);
		aabb.maxZ = std::max(aabb.maxZ, v.z);
	}
	return aabb;
}



std::vector<shared_ptr<Mesh>> Mesh::LoadObj(const std::string& fileName) {
	const std::string mtlDir = fileName.substr(0, fileName.rfind('/'));

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	bool readOK
		= tinyobj::LoadObj(&attributes,
			&shapes,
			&materials,
			&err,
			&err,
			fileName.data(),
			mtlDir.data());
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + fileName + ":" + mtlDir + " : " + err);
	}

	// if (materials.empty())
//		throw std::runtime_error("could not parse materials ...");

	vector<shared_ptr<Mesh>> meshes;

	std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t& shape = shapes[shapeID];

		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);

		for (int materialID : materialIDs) {
			std::map<tinyobj::index_t, int> knownVertices;
			auto mesh = std::make_shared<Mesh>();

			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

				int3 idx = make_int3(mesh->AddVertex(attributes, idx0, knownVertices),
					mesh->AddVertex(attributes, idx1, knownVertices),
					mesh->AddVertex(attributes, idx2, knownVertices));
				mesh->index.push_back(idx);
			}

			if (!mesh->vertex.empty()) {
				meshes.push_back(std::move(mesh));
			}
				
		}
	}

	std::cout << "created a total of " << meshes.size() << " meshes" << std::endl;
	return meshes;
}

std::vector<std::shared_ptr<Mesh>> Mesh::LoadObjPBR(const std::string& fileName, const Shader& pbrShader, const DisneyPbrData& dataTemplate) {
	const std::string mtlDir = fileName.substr(0, fileName.rfind('/'));

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::vector<shared_ptr<HostMaterial>> hostMaterials;
	std::string err = "";

	bool readOK = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err, fileName.data(), mtlDir.data());
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + fileName + ":" + mtlDir + " : " + err);
	}

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	
	texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 1;
	texDesc.maxMipmapLevelClamp = 99;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	texDesc.borderColor[0] = 1.0f;
	texDesc.sRGB = true;

	hostMaterials.resize(materials.size());

	for (int i = 0; i < materials.size(); i++) {
		const tinyobj::material_t& mat = materials[i];
		auto data = dataTemplate;

		if (!mat.diffuse_texname.empty()) {
			// TODO : Memory Manage
			texDesc.sRGB = true;
			auto texture = sutil::loadTexture(mat.diffuse_texname.c_str(), make_float3(1), &texDesc);
			data.baseColorTexture = texture.texture;
		}

		if (!mat.roughness_texname.empty()) {
			// TODO : Memory Manage
			texDesc.sRGB = false;
			auto texture = sutil::loadTexture(mat.roughness_texname.c_str(), make_float3(1), &texDesc);
			data.roughnessTexture = texture.texture;
		}

		if (!mat.metallic_texname.empty()) {
			// TODO : Memory Manage
			texDesc.sRGB = false;
			auto texture = sutil::loadTexture(mat.metallic_texname.c_str(), make_float3(1), &texDesc);
			data.metallicTexture = texture.texture;
		}

		if (!mat.normal_texname.empty()) {
			// TODO : Memory Manage
			texDesc.sRGB = false;
			auto texture = sutil::loadTexture(mat.normal_texname.c_str(), make_float3(1), &texDesc);
			data.normalTexture = texture.texture;
		}

		hostMaterials[i] = pbrShader.CreateHostMaterial<DisneyPbrData>(data);
	}

	vector<shared_ptr<Mesh>> meshes;

	std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t& shape = shapes[shapeID];

		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);

		for (int materialID : materialIDs) {
			std::map<tinyobj::index_t, int> knownVertices;
			auto mesh = std::make_shared<Mesh>();

			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

				int3 idx = make_int3(mesh->AddVertex(attributes, idx0, knownVertices),
					mesh->AddVertex(attributes, idx1, knownVertices),
					mesh->AddVertex(attributes, idx2, knownVertices));
				mesh->index.push_back(idx);
			}

			if (!mesh->vertex.empty()) {
				mesh->material = hostMaterials[materialID];
				meshes.push_back(std::move(mesh));
			}

		}
	}

	std::cout << "created a total of " << meshes.size() << " meshes" << std::endl;
	return meshes;
}


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

	index.push_back(firstVertexId + make_int3(0, 2, 1));
	index.push_back(firstVertexId + make_int3(0, 3, 2));

	index.push_back(firstVertexId + make_int3(2, 3, 5));
	index.push_back(firstVertexId + make_int3(2, 5, 4));

	index.push_back(firstVertexId + make_int3(4, 5, 6));
	index.push_back(firstVertexId + make_int3(7, 6, 5));

	index.push_back(firstVertexId + make_int3(0, 1, 6));
	index.push_back(firstVertexId + make_int3(0, 6, 7));

	index.push_back(firstVertexId + make_int3(1, 2, 6));
	index.push_back(firstVertexId + make_int3(2, 4, 6));

	index.push_back(firstVertexId + make_int3(0, 7, 3));
	index.push_back(firstVertexId + make_int3(3, 7, 5));
}

void Mesh::AddToplessCube(float3& center, const float3& halfSize) {
	int3 firstVertexId = make_int3((int)vertex.size());

	vertex.push_back(center + make_float3(-1, -1, -1) * halfSize); // 0
	vertex.push_back(center + make_float3(-1, 1, -1) * halfSize); // 1
	vertex.push_back(center + make_float3(-1, 1, 1) * halfSize); // 2
	vertex.push_back(center + make_float3(-1, -1, 1) * halfSize); // 3
	vertex.push_back(center + make_float3(1, 1, 1) * halfSize); // 4
	vertex.push_back(center + make_float3(1, -1, 1) * halfSize); // 5
	vertex.push_back(center + make_float3(1, 1, -1) * halfSize); // 6
	vertex.push_back(center + make_float3(1, -1, -1) * halfSize); // 7

	index.push_back(firstVertexId + make_int3(0, 2, 1));
	index.push_back(firstVertexId + make_int3(0, 3, 2));

	index.push_back(firstVertexId + make_int3(2, 3, 5));
	index.push_back(firstVertexId + make_int3(2, 5, 4));

	index.push_back(firstVertexId + make_int3(4, 5, 6));
	index.push_back(firstVertexId + make_int3(7, 6, 5));

	index.push_back(firstVertexId + make_int3(0, 1, 6));
	index.push_back(firstVertexId + make_int3(0, 6, 7));

	index.push_back(firstVertexId + make_int3(0, 7, 3));
	index.push_back(firstVertexId + make_int3(3, 7, 5));
}

void Mesh::AddTriangle(const float3& a, const float3& b, const float3& c) {
	int3 firstVertexId = make_int3((int)vertex.size());

	vertex.push_back(a);
	vertex.push_back(b);
	vertex.push_back(c);

	index.push_back(firstVertexId + make_int3(0, 1, 2));
}

void Mesh::AddPrism(const float3& center, const float halfsize, const float3& vertix)
{
	int3 firstVertexId = make_int3((int)vertex.size());

	vertex.push_back(center + make_float3(0, halfsize, halfsize ));
	vertex.push_back(center + make_float3(0,halfsize, -halfsize));
	vertex.push_back(center + make_float3(0, -halfsize, halfsize ));
	vertex.push_back(center + make_float3(0, -halfsize, -halfsize));
	vertex.push_back(vertix);

	index.push_back(firstVertexId + make_int3(0, 2, 1));
	index.push_back(firstVertexId + make_int3(1, 2, 3));
	index.push_back(firstVertexId + make_int3(0, 1, 4));
	index.push_back(firstVertexId + make_int3(1, 3, 4));
	index.push_back(firstVertexId + make_int3(2, 0, 4));
	index.push_back(firstVertexId + make_int3(3, 2, 4));
}



void Mesh::Rotate(float angle, const float3& axis) {
	auto matrix = sutil::Matrix<4, 4>::rotate(angle * M_PI / 180, axis);

	for (auto& v : vertex) {
		auto v4 = make_float4(v, 1.f) * matrix;
		v = make_float3(v4);
	}
}

void Mesh::Move(const float3& delta) {
	for (auto& v : vertex) {
		v += delta;
	}
}

void Mesh::Scale(const float3& scale) {
	for (auto& v : vertex) {
		v *= scale;
	}
}

void Mesh::GetBuildInput(OptixBuildInput& input) {
	deviceVertex.Upload(vertex);
	deviceIndex.Upload(index);
	if (normal.size() > 0) {
		deviceNormal.Upload(normal);
	}
	if (texcoord.size() > 0) {
		deviceTexcoord.Upload(texcoord);
	}

	input = {};
	input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	OptixBuildInputTriangleArray& tArray = input.triangleArray;


	tArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	tArray.vertexStrideInBytes = sizeof(float3);
	tArray.numVertices = (int)vertex.size();
	tArray.vertexBuffers = deviceVertex.GetDevicePointerRef();

	tArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	tArray.indexStrideInBytes = sizeof(int3);
	tArray.numIndexTriplets = (int)index.size();
	tArray.indexBuffer = deviceIndex.GetDevicePointer();

	tArray.flags = &geometryFlags;
	tArray.numSbtRecords = 1;
	tArray.sbtIndexOffsetBuffer = NULL;

}

void Mesh::GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) {
	DeviceMeshData& data = record.data.data.mesh;
	// Object Type 0 For Mesh
	CheckOptiXErrors(optixSbtRecordPackHeader(hitPrograms[MESH_OBJECT_TYPE], &record));

	data.vertex = (float3*)deviceVertex.GetDevicePointer();
	data.index = (int3*)deviceIndex.GetDevicePointer();
	data.normal = normal.size() > 0 ? (float3*)deviceNormal.GetDevicePointer() : nullptr;
	data.texcoord = texcoord.size() > 0 ? (float2*)deviceTexcoord.GetDevicePointer() : nullptr;
	record.data.directLightId = directLightId;
	record.data.material = material->CreateMaterial();
}



void Sphere::GetBuildInput(OptixBuildInput& input) {
	devicePosition.Upload(&position, 1);
	deviceRadius.Upload(&radius, 1);

	input = {};
	input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

	OptixBuildInputSphereArray& sArray = input.sphereArray;
	
	sArray.radiusBuffers = deviceRadius.GetDevicePointerRef();

	sArray.vertexBuffers = devicePosition.GetDevicePointerRef();
	sArray.numVertices = 1;

	sArray.flags = &geometryFlags;
	sArray.numSbtRecords = 1;
	sArray.sbtIndexOffsetBuffer = NULL;
}

void Sphere::GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) {
	DeviceSphereData& data = record.data.data.sphere;
	// Object Type 1 For Sphere
	CheckOptiXErrors(optixSbtRecordPackHeader(hitPrograms[SPHERE_OBJECT_TYPE], &record));

	data.position = position;
	data.radius = radius;
	record.data.directLightId = directLightId;
	record.data.material = material->CreateMaterial();
}

void Curve::GetBuildInput(OptixBuildInput& input) {

	CalculateComb();
	aabb = GetAabb();

	devicePoints.Upload(points);
	deviceAabb.Upload(&aabb, 1);
	deviceCombs.Upload(combs);

	input = {};
	input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

	OptixBuildInputCustomPrimitiveArray& cArray = input.customPrimitiveArray;

	cArray.numPrimitives = 1;
	cArray.aabbBuffers = deviceAabb.GetDevicePointerRef();
	cArray.flags = &geometryFlags;

	cArray.numSbtRecords = 1;
	cArray.sbtIndexOffsetBuffer = NULL;
}

void Curve::GetShaderBindingRecord(HitgroupRecord& record, const std::vector<OptixProgramGroup>& hitPrograms) {
	DeviceCurveData& data = record.data.data.curve;
	// Object Type 2 For Curve
	CheckOptiXErrors(optixSbtRecordPackHeader(hitPrograms[CURVE_OBJECT_TYPE], &record));

	data.n = points.size()-1;
	data.aabb = (OptixAabb*)deviceAabb.GetDevicePointer();
	data.theta = theta;
	data.points = (float3*)devicePoints.GetDevicePointer();
	data.position = position;
	data.combs = (int*)deviceCombs.GetDevicePointer();
	record.data.directLightId = directLightId;
	record.data.material = material->CreateMaterial();
}

OptixAabb Curve::GetAabb() {
	float3 min_range = make_float3(1e10);
	float3 max_range = make_float3(-1e10);

	GetRange(min_range, max_range);

	return OptixAabb{
		/* minX = */ min_range.x, /* minY = */ min_range.y, /* minZ = */ min_range.z,
		/* maxX = */ max_range.x, /* maxY = */ max_range.y, /* maxZ = */ max_range.z
	};
}

void Curve::CalculateComb() {
	int n = points.size()-1;
	combs.resize(n+1);
	int cni = 1;
	for (int i = 0; i <= n; i++) {
		combs[i] = cni;
		cni = cni * (n - i) / (i + 1);
	}
}

float Curve::CalculateX(float v) {
	int n = points.size()-1;
	float result = 0.f;
	for (int i = 0; i <= n; i++) {
		result += points[i].x * combs[i] * pow(v, i) * pow(1 - v, n - i);
	}
	return result;
}

float Curve::CalculateY(float v)
{
	int n = points.size()-1;
	float result = 0.f;
	for (int i = 0; i <= n; i++) {
		result += points[i].y * combs[i] * pow(v, i) * pow(1 - v, n - i);
	}
	return result;
}

void Curve::ZRotate(float3& vec)
{
	float tmpx = vec.x * cos(theta) - vec.y * sin(theta);
	float tmpy = vec.x * sin(theta) + vec.y * cos(theta);
	vec.x = tmpx;
	vec.y = tmpy;
}

void Curve::GetRange(float3& min_range, float3& max_range)
{
	int N = 100;
	for (int i = 0; i <= N; i++) {
		for (int j = 0; j <= N; j++) {
			float u = 2 * M_PI / N * j;
			float tmpx = CalculateX((float)i / N);
			float tmpy = CalculateY((float)i / N);

			float3 tmp = { tmpx * cos(u), tmpy, -tmpx * sin(u) };
			ZRotate(tmp);
			tmp += position;

			if (tmp.x < min_range.x) {
				min_range.x = tmp.x;
			}

			if (tmp.y < min_range.y) {
				min_range.y = tmp.y;
			}

			if (tmp.z < min_range.z) {
				min_range.z = tmp.z;
			}

			if (tmp.x > max_range.x) {
				max_range.x = tmp.x;
			}

			if (tmp.y > max_range.y) {
				max_range.y = tmp.y;
			}

			if (tmp.z > max_range.z) {
				max_range.z = tmp.z;
			}

		}
	}

}

