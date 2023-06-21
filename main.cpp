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
        
        PathTracer renderer;

        Shader& naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");

        Shader& naiveMirrorShader = renderer.CreateShader("__direct_callable__naive_mirror");

        NaviveDiffuseData white, whiteLight, red, green;
    
        white.albedo = make_float3(0.8);
        whiteLight.emission = make_float3(6, 6, 2);
        red.albedo = make_float3(0.8f, 0.05f, 0.05f);
        green.albedo = make_float3(0.05f, 0.8f, 0.05f);

        NaiveMirrorData mirror;

        auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
        auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
        auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
        auto blueMat = naiveDiffuseShader.CreateHostMaterial(green);

        auto mirrorMat = naiveMirrorShader.CreateHostMaterial(mirror);

        Mesh plane, cube, leftPlane, rightPlane, light, cube2;

        plane.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
        plane.AddCube(make_float3(0, -3, 0), make_float3(3, 0.1f, 3));
        plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
        plane.material = whiteMat;

        leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
        leftPlane.material = blueMat;

        rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
        rightPlane.material = redMat;

        cube2.AddCube(make_float3(0, 0, 0), make_float3(0.75f, 0.75F, 0.75f));
        cube2.Rotate(-30, make_float3(0, 1, 0));
        cube2.Move(make_float3(1, -2, -1));
        cube2.material = whiteMat;

        cube.AddCube(make_float3(0, 0, 0), make_float3(0.75f, 2.f, 1.f));
        cube.Rotate(30, make_float3(0, 1, 0));
        cube.Move(make_float3(-1, -1, 1.5f));
        cube.material = mirrorMat;

        light.AddCube(make_float3(0, 2.5, 0), make_float3(1, 0.2, 1));
        light.material = whiteLightMat;
        
        Sphere haha;

        haha.position = make_float3(2, 0, 0);
        haha.radius = 0.5f;
        // haha.material.albedo = make_float3(1.f, 1.f, 1.f);

        Sphere test;
        test.position = make_float3(-1, 0, 0);
        test.radius = 1.f;
        // test.material.emission = make_float3(1.f, 1.f, 1.f);

       

        renderer.AddMesh(std::move(plane));
        renderer.AddMesh(std::move(cube));
        renderer.AddMesh(std::move(cube2));
        renderer.AddMesh(std::move(leftPlane));
        renderer.AddMesh(std::move(rightPlane));
        renderer.AddMesh(std::move(light));

        // renderer.AddSphere(std::move(haha));
        // renderer.AddSphere(std::move(test));

        PathTracerWindow window(&renderer);

        window.Run();
    }
    catch (std::runtime_error& e) {
        
        exit(1);
    }


    return 0;
}