#include "optixLib.h"
#include "sutil\sutil.h"
#include "mesh.h"
#include "renderer.h"
#include "window.h"

#include "sutil/CUDAOutputBuffer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <support\tinygltf\stb_image_write.h>

extern "C" int main(int ac, char** av) {
    
    try {
        InitCudaAndOptix();
        
        Mesh plane, cube, leftPlane, rightPlane, light;


        plane.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
        plane.AddCube(make_float3(0, -3, 0), make_float3(3, 0.1f, 3));
        plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));

        plane.material.albedo = make_float3(1.f, 1.f, 1.f);

        leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
        leftPlane.material.albedo = make_float3(1, 0, 0);

        rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
        rightPlane.material.albedo = make_float3(0, 0, 1);

        cube.AddCube(make_float3(-1.5, -1, 0), make_float3(0.75f, 2.f, 0.5f));
        cube.Rotate(30, make_float3(0, 1, 0));
        cube.material.emission = make_float3(0, 1.f, 0);
        cube.material.albedo = make_float3(1.f, 1.f, 1.f);

        light.AddCube(make_float3(0, 2.5, 0), make_float3(1, 0.2, 1));
        light.material.emission = make_float3(1, 1, 1);
        

        PathTracer renderer;


        renderer.AddMesh(std::move(plane));
        renderer.AddMesh(std::move(cube));
        renderer.AddMesh(std::move(leftPlane));
        renderer.AddMesh(std::move(rightPlane));
        renderer.AddMesh(std::move(light));

        PathTracerWindow window(&renderer);

        window.Run();
    }
    catch (std::runtime_error& e) {
        
        exit(1);
    }


    return 0;
}