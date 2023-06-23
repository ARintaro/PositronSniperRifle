#pragma once
#include <string>
#include "cudaBuffer.h";
#include <memory>
#include "renderParams.h"


struct HostMaterial {
	std::unique_ptr<CudaBuffer> data;
	float3 emission;
	int programIndex;

	template<typename DataType>
	HostMaterial(DataType _data, int _programIndex, float3 _emission) {
		data = std::make_unique<CudaBuffer>();
		data->Upload(&_data, 1);
		programIndex = _programIndex;
		emission = _emission;
	}

	Material CreateMaterial() {
		Material material;
		material.data = data->GetDevicePointerVoid();
		material.programIndex = programIndex;
		material.emission = emission;
		return material;
	}

	~HostMaterial() {
		if (data) {
			data->Free();
		}
	}

};

struct Shader {
	std::string name;
	int programIndex;
	OptixProgramGroup program = nullptr;

	template<typename DataType>
	std::shared_ptr<HostMaterial> CreateHostMaterial(DataType _data, float3 emission = make_float3(0)) {
		return std::make_shared<HostMaterial>(_data, programIndex, emission);
	}
};
