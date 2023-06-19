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
        
        Mesh plane, cube, leftPlane, rightPlane;

        plane.color = make_float3(0.1f, 0.1f, 0.1f);
        cube.color = make_float3(0, 0, 0);
        leftPlane.color = make_float3(1, 0, 0);
        rightPlane.color = make_float3(0, 0, 1);

        plane.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
        plane.AddCube(make_float3(0, -3, 0), make_float3(3, 0.1f, 3));
        plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));

        plane.material.programIndex = 0;

        leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));

        leftPlane.material.programIndex = 0;

        rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));

        rightPlane.material.programIndex = 0;

        cube.AddCube(make_float3(-1.5, -1, 0), make_float3(0.75f, 2.f, 0.5f));

        cube.Rotate(30, make_float3(0, 1, 0));

        cube.material.programIndex = 1;

        PathTracer renderer;


        renderer.AddMesh(std::move(plane));
        renderer.AddMesh(std::move(cube));
        renderer.AddMesh(std::move(leftPlane));
        renderer.AddMesh(std::move(rightPlane));

        PathTracerWindow window(&renderer);

        window.Run();
    }
    catch (std::runtime_error& e) {
        
        exit(1);
    }


    return 0;
}