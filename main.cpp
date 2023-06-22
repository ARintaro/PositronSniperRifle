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

        Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
        Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
        Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
        Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");

        NaiveDiffuseData white, whiteLight, red, green;
    
        white.albedo = make_float3(0.8);
        whiteLight.emission = make_float3(6, 6, 2);
        red.albedo = make_float3(0.8f, 0.05f, 0.05f);
        green.albedo = make_float3(0.05f, 0.8f, 0.05f);

        NaiveMetalData metal;

        metal.roughness = 0.5f;

        auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
        auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
        auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
        auto blueMat = naiveDiffuseShader.CreateHostMaterial(green);
        auto mirrorMat = naiveMetalShader.CreateHostMaterial(metal);

        
        NaiveDielectricsData dielectrics;
        dielectrics.refractivity = 1.5f;

        auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);


        DisneyPbrData pbrData;
        pbrData.baseColor = make_float3(0.2f, 0.8f, 0.2f);
        pbrData.roughness = 0.3f;
        pbrData.specular = 0.2f;
        pbrData.subsurface = 1.0f;
        pbrData.specularTint = 0.6f;
        pbrData.metallic = 0.7f;

        auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

        Mesh plane, cube, leftPlane, rightPlane, light, cube2;

        // plane.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
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

        haha.position = make_float3(0, 0, 0);
        haha.radius = 1.5f;
        haha.material = dielectricsMat;

        Sphere test;
        test.position = make_float3(-1, 0, 1);
        test.radius = 1.5f;
        // test.material.emission = make_float3(1.f, 1.f, 1.f);
        test.material = mirrorMat;

        Curve curve;

        curve.position = make_float3(0, -1, 1);
        curve.points = { make_float3(1, -1, 0), make_float3(2, 0, 0) , make_float3(1, 1, 0)};
        curve.theta = 0;
        curve.material = mirrorMat;

        Mesh bunny = Mesh::LoadObj("../models/bunny.obj")[0];
        bunny.material = whiteMat;
        bunny.Scale(make_float3(1.5f));
        bunny.Move(make_float3(0.5f, -1.0f, 1.f));
        
             
        renderer.AddCurve(std::move(curve));
        // renderer.AddMesh(std::move(bunny));
        renderer.AddMesh(std::move(plane));
        // renderer.AddMesh(std::move(cube));
        // renderer.AddMesh(std::move(cube2));
        renderer.AddMesh(std::move(leftPlane));
        renderer.AddMesh(std::move(rightPlane));
        renderer.AddMesh(std::move(light));

        // renderer.AddSphere(std::move(haha));
        // renderer.AddSphere(std::move(test));

        PathTracerWindow window(&renderer);

        window.Run();
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }


    return 0;
}