#pragma once
#include <string>
#include "cudaBuffer.h";
#include <memory>
#include "renderParams.h"


struct HostMaterial {
	std::unique_ptr<CudaBuffer> data;
	int programIndex;

	template<typename DataType>
	HostMaterial(DataType _data, int _programIndex) {
		data = std::make_unique<CudaBuffer>();
		data->Upload(&_data, 1);
		programIndex = _programIndex;
	}

	Material CreateMaterial() {
		Material material;
		material.data = data->GetDevicePointerVoid();
		material.programIndex = programIndex;
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
	std::shared_ptr<HostMaterial> CreateHostMaterial(DataType _data) {
		return std::make_shared<HostMaterial>(_data, programIndex);
	}
};
