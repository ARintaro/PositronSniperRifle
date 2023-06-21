#pragma once

#include "optixLib.h"
#include <assert.h>
#include <vector>

class CudaBuffer {

public:
	constexpr inline CUdeviceptr GetDevicePointer() { return (CUdeviceptr)devicePointer; }
	constexpr inline void* GetDevicePointerVoid() { return devicePointer; }
	constexpr inline CUdeviceptr* GetDevicePointerRef() { return (CUdeviceptr*)(&devicePointer); }
	constexpr inline size_t GetSize() { return size; }

	void Resize(size_t size) {
		if (this->devicePointer) {
			Free();
		}
		Alloc(size);
	}

	void Alloc(size_t size) {
		assert(devicePointer == nullptr);
		this->size = size;
		
		CheckCudaErrors(cudaMalloc((void**)&devicePointer, size));
	}

	void Free() {
		assert(devicePointer != nullptr);
		CheckCudaErrors(cudaFree(devicePointer));
		devicePointer = nullptr;
		size = 0;
	}
	 
	template<typename T>
	void Upload(const T* hostPointer, size_t count) {
		size_t dataSize = count * sizeof(T);
		if (devicePointer == nullptr) {
			Alloc(dataSize);
		}
		assert(size == dataSize);
		CheckCudaErrors(cudaMemcpy(devicePointer, (void*)hostPointer, dataSize, cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Upload(const std::vector<T>& data) {
		Upload(data.data(), data.size());
	}

	template<typename T>
	void Download(T* hostPointer, size_t count) {
		size_t dataSize = count * sizeof(T);
		assert(dataSize == size && devicePointer != nullptr);
		CheckCudaErrors(cudaMemcpy((void*)hostPointer, devicePointer, dataSize, cudaMemcpyDeviceToHost));
	}

private:
	size_t size = 0;
	void* devicePointer = nullptr;

};