#include "optixLib.h"
#include "sutil\sutil.h"
#include "mesh.h"
#include "renderer.h"
#include "window.h"

#include "sutil/CUDAOutputBuffer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <support\tinygltf\stb_image_write.h>

//static void Scene1() {
//    PathTracer renderer;
//
//    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
//    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
//    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
//    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
//    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");
//
//    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;
//
//    white.albedo = make_float3(0.8);
//    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
//    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
//    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
//    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);
//
//    NaiveMetalData metal;
//
//    metal.roughness = 0.f;
//
//    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
//    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
//    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
//    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
//    auto blueMat = naiveDiffuseShader.CreateHostMaterial(blue);
//    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
//    auto mirrorMat = naiveMetalShader.CreateHostMaterial(metal);
//    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);
//
//    NaiveDielectricsData dielectrics;
//    dielectrics.refractivity = 1.5f;
//
//    auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);
//
//
//    DisneyPbrData pbrData;
//    pbrData.baseColor = make_float3(0.2f, 0.2f, 0.8f);
//    pbrData.roughness = 0.5f;
//    pbrData.specular = 0.2f;
//    pbrData.subsurface = 1.0f;
//    pbrData.specularTint = 0.6f;
//    pbrData.metallic = 0.5f;
//
//    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);
//
//    Mesh plane, leftPlane, rightPlane, light, back;
//
//    plane->AddCube(make_float3(0, -3, 0), make_float3(6, 0.1f, 6));
//    plane->AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
//    plane.material = whiteMat;
//
//    back->AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
//    back.material = lightyellowMat;
//
//    leftPlane->AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
//    leftPlane.material = redMat;
//
//    rightPlane->AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
//    rightPlane.material = greenMat;
//
//    light->AddCube(make_float3(0.f, 2.5f, 0.f), make_float3(1.f, 0.2f, 1.f));
//    light.material = whiteLightMat;
//
//    Mesh cube;
//    cube->AddCube(make_float3(0.8f, -1.8f, -1.5f), make_float3(0.5f, 0.5f, 0.5f));
//    cube.Rotate(50, make_float3(0.f, 1.f, 0.f));
//    cube.material = mirrorMat;
//
//    Curve curve;
//
//    curve.position = make_float3(-1.5f, -3.f, -1.f);
//    curve.points = { make_float3(0.5f, 0.f, 0), make_float3(1.f, 0.5f, 0) , make_float3(0.5f, 1.f, 0), make_float3(0.25f, 1.5f, 0), make_float3(0.5f, 2.f, 0) };
//    curve.theta = 0;
//    curve.material = roughPbr;
//
//    Mesh bunny = Mesh::LoadObj("../models/bunny.obj")[0];
//    bunny.material = dielectricsMat;
//    bunny.Scale(make_float3(1.5f));
//    bunny.Move(make_float3(0.5f, -1.f, -4));
//
//    renderer.AddMesh(std::move(leftPlane));
//    renderer.AddMesh(std::move(rightPlane));
//    renderer.AddMesh(std::move(plane));
//    renderer.AddMesh(std::move(back));
//    renderer.AddMesh(std::move(light));
//    renderer.AddMesh(std::move(bunny));
//    renderer.AddCurve(std::move(curve));
//    renderer.AddMesh(std::move(cube));
//
//    PathTracerWindow window(&renderer);
//
//    window.Run();
//}

static void Scene2() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;

    white.albedo = make_float3(0.8);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);
    white.albedo = make_float3(0.8);

    NaiveMetalData metal;

    metal.roughness = 0.4f;

    NaiveDielectricsData dielectrics;
    dielectrics.refractivity = 1.5f;


    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight, make_float3(4, 4, 2));
    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);
    auto mirrorMat = naiveMetalShader.CreateHostMaterial(metal);
    auto glass = naiveDielectrixShader.CreateHostMaterial(dielectrics);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.1f, 0.1f, 0.6f);
    pbrData.roughness = 0.3f;
    pbrData.specular = 0.3f;
    pbrData.subsurface = 0.8f;
    pbrData.specularTint = 0.3f;
    pbrData.metallic = 0.7f;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    auto plane = std::make_shared<Mesh>();
    auto leftPlane = std::make_shared<Mesh>(), rightPlane = std::make_shared<Mesh>() , light = std::make_shared<Mesh>(), back = std::make_shared<Mesh>();

    plane->AddCube(make_float3(0, -3, 0), make_float3(6, 0.1f, 6));
    // plane->AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    plane->material = whiteMat;

    back->AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
    back->material = lightyellowMat;

    leftPlane->AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
    leftPlane->material = redMat;

    rightPlane->AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
    rightPlane->material = greenMat;

    light->AddCube(make_float3(0, 2, 0), make_float3(1.f, 0.2, 1.f));
    light->material = whiteLightMat;

    renderer.SetDirectLight(*light);
    

    auto bunny = Mesh::LoadObj("../models/bunny.obj")[0];
    bunny->material = redMat;
    bunny->Scale(make_float3(1.5f));
    bunny->Move(make_float3(1.5f, -2.5, -3));

    auto sphere = std::make_shared<Sphere>();
    sphere->position = make_float3(-1.f, -1.f, 1.f);
    sphere->radius = 1.f;
    sphere->material = roughPbr;

    renderer.AddMesh(std::move(leftPlane));
    renderer.AddMesh(std::move(rightPlane));
    renderer.AddMesh(std::move(plane));
    renderer.AddMesh(std::move(back));
    renderer.AddMesh(std::move(light));
    renderer.AddMesh(std::move(bunny));
    renderer.AddSphere(std::move(sphere));

    PathTracerWindow window(&renderer);

    window.Run();
}

static void Scene3() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;

    NaiveDielectricsData dielectrics;
    dielectrics.refractivity = 1.5f;

    auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);

    white.albedo = make_float3(0.5);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);
    white.albedo = make_float3(0.8);

    NaiveMetalData metal;

    metal.roughness = 0.3f;

    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight, make_float3(4, 4, 2) * 10);
    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);
    auto mirrorMat = naiveDielectrixShader.CreateHostMaterial(metal);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.1f, 0.1f, 0.6f);
    pbrData.roughness = 0.1f;
    pbrData.specular = 0.3f;
    pbrData.subsurface = 0.8f;
    pbrData.specularTint = 0.3f;
    pbrData.metallic = 1;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    auto plane = std::make_shared<Mesh>();
    auto leftPlane = std::make_shared<Mesh>(), rightPlane = std::make_shared<Mesh>(), light = std::make_shared<Mesh>(), back = std::make_shared<Mesh>();

    plane->AddCube(make_float3(0, -3, 0), make_float3(3, 0.1f, 3));
    plane->AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    plane->AddCube(make_float3(3, 0, 0), make_float3(0.01f, 3, 3));
    plane->AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
    plane->AddCube(make_float3(-3, -1, 0), make_float3(0.01f, 2.f, 3));
    plane->AddCube(make_float3(-6, 0, -3), make_float3(3, 3, 1));

    // rightPlane->AddCube(make_float3(-3, 1, 0), make_float3(0.1f, 2.f, 3));
    rightPlane->AddCube(make_float3(-3, 3, 0), make_float3(0.005f, 1.8f, 3));
    rightPlane->AddCube(make_float3(-3, 1.2f, 3), make_float3(0.005f, 2.f, 2.8f));
    rightPlane->AddCube(make_float3(-3, 1.2f, -3), make_float3(0.005f, 2.f, 2.8f));

    rightPlane->material = whiteMat;

    // plane->AddCube(make_float3(0, 0, 0), make_float3(300, 300, 300));
    plane->material = mirrorMat;


    light->AddCube(make_float3(-6, 2.f, 0), make_float3(0.2f, 0.1f, 0.1f));
    light->material = whiteLightMat;

    auto bunny = Mesh::LoadObj("../models/bunny.obj")[0];
    bunny->material = dispersionMat;
    bunny->Scale(make_float3(1.5f));
    bunny->Move(make_float3(1.5f, -2.5, -3));

    auto sphere = std::make_shared<Sphere>();
    sphere->position = make_float3(-1.f, -1.f, 1.f);
    sphere->radius = 1.f;
    sphere->material = whiteMat;

    renderer.renderParams.globalFogDensity = 0.05f;
    renderer.renderParams.maxDepth = 3;

    renderer.SetDirectLight(*light);

    renderer.AddMesh(std::move(plane));
    renderer.AddMesh(std::move(light));
    renderer.AddMesh(std::move(bunny));
    renderer.AddSphere(std::move(sphere));
    renderer.AddMesh(std::move(rightPlane));

    PathTracerWindow window(&renderer);

    window.Run();
}

static void Scene4() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;

    NaiveDielectricsData dielectrics;
    dielectrics.refractivity = 2.f;

    auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);

    white.albedo = make_float3(0.5);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);
    white.albedo = make_float3(0.8);

    NaiveMetalData metal;

    metal.roughness = 0.5f;

    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight, make_float3(1, 1, 0.8));
    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);
    auto mirrorMat = naiveDielectrixShader.CreateHostMaterial(metal);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.6f, 0.6f, 0.6f);
    pbrData.roughness = 0.3f;
    pbrData.specular = 0.3f;
    pbrData.subsurface = 0.8f;
    pbrData.specularTint = 0.3f;
    pbrData.metallic = 0.5f;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    auto plane = std::make_shared<Mesh>();
    auto leftPlane = std::make_shared<Mesh>(), rightPlane = std::make_shared<Mesh>(), light = std::make_shared<Mesh>(), back = std::make_shared<Mesh>();

    plane->AddCube(make_float3(0, -3, 0), make_float3(3, 0.1f, 3));
    plane->AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    rightPlane->AddCube(make_float3(3, 0, 0), make_float3(0.01f, 3, 3));
    leftPlane->AddCube(make_float3(-3, 0, 0), make_float3(0.01f, 3, 3));
    plane->AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.01f));

    rightPlane->material = redMat;
    leftPlane->material = greenMat;

    // plane->AddCube(make_float3(0, 0, 0), make_float3(300, 300, 300));
    plane->material = whiteMat;


    light->AddCube(make_float3(0, 2, 0), make_float3(1.f, 0.2, 1.f));
    light->material = whiteLightMat;

    auto water = Mesh::LoadObj("../models/water.obj")[0];
    water->material = dielectricsMat;
    water->Scale(make_float3(1.5f));
    water->Move(make_float3(-3.5f, -3.f, -3.f));
    water->AddToplessCube(make_float3(0.f, -2.f, 0.f), make_float3(3.f, 1.5f, 3.f));

    

    // water->Scale(make_float3(0.5f));


    // renderer.renderParams.globalFogDensity = 0.02f;
    // renderer.renderParams.maxDepth = 3;

    // renderer.SetDirectLight(*light);

    renderer.AddMesh(std::move(plane));
    renderer.AddMesh(std::move(light));
    renderer.AddMesh(std::move(water));
    renderer.AddMesh(std::move(leftPlane));
    renderer.AddMesh(std::move(rightPlane));
    // renderer.AddMesh(std::move(bunny));

    PathTracerWindow window(&renderer);

    window.Run();

}

extern "C" int main(int ac, char** av) {
    
    try {
        InitCudaAndOptix();
        Scene4();
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }


    return 0;
}