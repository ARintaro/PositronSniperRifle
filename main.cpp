#include "optixLib.h"
#include "sutil\sutil.h"
#include "mesh.h"
#include "renderer.h"
#include "window.h"

#include "sutil/CUDAOutputBuffer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <support\tinygltf\stb_image_write.h>

static void Scene1() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;

    white.albedo = make_float3(0.8);
    whiteLight.emission = make_float3(9, 9, 3);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);

    NaiveMetalData metal;

    metal.roughness = 0.f;

    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto blueMat = naiveDiffuseShader.CreateHostMaterial(blue);
    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
    auto mirrorMat = naiveMetalShader.CreateHostMaterial(metal);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);

    NaiveDielectricsData dielectrics;
    dielectrics.refractivity = 1.5f;

    auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.2f, 0.2f, 0.8f);
    pbrData.roughness = 0.5f;
    pbrData.specular = 0.2f;
    pbrData.subsurface = 1.0f;
    pbrData.specularTint = 0.6f;
    pbrData.metallic = 0.5f;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    Mesh plane, leftPlane, rightPlane, light, back;

    plane.AddCube(make_float3(0, -3, 0), make_float3(6, 0.1f, 6));
    plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    plane.material = whiteMat;

    back.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
    back.material = lightyellowMat;

    leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
    leftPlane.material = redMat;

    rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
    rightPlane.material = greenMat;

    light.AddCube(make_float3(0.f, 2.5f, 0.f), make_float3(1.f, 0.2f, 1.f));
    light.material = whiteLightMat;

    Mesh cube;
    cube.AddCube(make_float3(0.8f, -1.8f, -1.5f), make_float3(0.5f, 0.5f, 0.5f));
    cube.Rotate(50, make_float3(0.f, 1.f, 0.f));
    cube.material = mirrorMat;

    Curve curve;

    curve.position = make_float3(-1.5f, -3.f, -1.f);
    curve.points = { make_float3(0.5f, 0.f, 0), make_float3(1.f, 0.5f, 0) , make_float3(0.5f, 1.f, 0), make_float3(0.25f, 1.5f, 0), make_float3(0.5f, 2.f, 0) };
    curve.theta = 0;
    curve.material = blueMat;

    Mesh bunny = Mesh::LoadObj("../models/bunny.obj")[0];
    bunny.material = dielectricsMat;
    bunny.Scale(make_float3(1.5f));
    bunny.Move(make_float3(0.5f, -1.f, -4));

    renderer.AddMesh(std::move(leftPlane));
    renderer.AddMesh(std::move(rightPlane));
    renderer.AddMesh(std::move(plane));
    renderer.AddMesh(std::move(back));
    renderer.AddMesh(std::move(light));
    renderer.AddMesh(std::move(bunny));
    renderer.AddCurve(std::move(curve));
    renderer.AddMesh(std::move(cube));

    PathTracerWindow window(&renderer);

    window.Run();
}

static void Scene2() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green, blue, lightyellow;

    white.albedo = make_float3(0.8);
    whiteLight.emission = make_float3(9, 9, 3);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);
    blue.albedo = make_float3(0.05f, 0.05f, 0.8f);
    lightyellow.albedo = make_float3(0.3f, 0.3f, 0.1f);
    white.albedo = make_float3(0.8);
    whiteLight.emission = make_float3(9, 9, 9);

    NaiveMetalData metal;

    metal.roughness = 0.f;

    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
    auto lightyellowMat = naiveDiffuseShader.CreateHostMaterial(lightyellow);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto greenMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.1f, 0.1f, 0.6f);
    pbrData.roughness = 0.5f;
    pbrData.specular = 0.2f;
    pbrData.subsurface = 1.0f;
    pbrData.specularTint = 0.6f;
    pbrData.metallic = 0.5f;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    Mesh plane, leftPlane, rightPlane, light, back;

    plane.AddCube(make_float3(0, -3, 0), make_float3(6, 0.1f, 6));
    plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    plane.material = whiteMat;

    back.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
    back.material = lightyellowMat;

    leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
    leftPlane.material = lightyellowMat;

    rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
    rightPlane.material = lightyellowMat;

    light.AddCube(make_float3(0, 2.5, 0), make_float3(1.f, 0.2, 1.f));
    light.material = whiteLightMat;

    Mesh bunny = Mesh::LoadObj("../models/bunny.obj")[0];
    bunny.material = dispersionMat;
    bunny.Scale(make_float3(1.5f));
    bunny.Move(make_float3(1.5f, -2.5, -3));

    Sphere sphere;
    sphere.position = make_float3(-1.f, -1.f, 1.f);
    sphere.radius = 1.f;
    sphere.material = roughPbr;

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


static void SampleSceneMirror() {
    PathTracer renderer;

    Shader naiveDiffuseShader = renderer.CreateShader("__direct_callable__naive_diffuse");
    Shader naiveDielectrixShader = renderer.CreateShader("__direct_callable__naive_dielectrics");
    Shader naiveMetalShader = renderer.CreateShader("__direct_callable__naive_metal");
    Shader disneyPbrShader = renderer.CreateShader("__direct_callable__disney_pbr");
    Shader dispersionShader = renderer.CreateShader("__direct_callable__dispersion");

    NaiveDiffuseData white, whiteLight, red, green;

    white.albedo = make_float3(0.8);
    whiteLight.emission = make_float3(6, 6, 6);
    red.albedo = make_float3(0.8f, 0.05f, 0.05f);
    green.albedo = make_float3(0.05f, 0.8f, 0.05f);

    NaiveMetalData metal;

    metal.roughness = 0.f;

    auto whiteMat = naiveDiffuseShader.CreateHostMaterial(white);
    auto whiteLightMat = naiveDiffuseShader.CreateHostMaterial(whiteLight);
    auto redMat = naiveDiffuseShader.CreateHostMaterial(red);
    auto blueMat = naiveDiffuseShader.CreateHostMaterial(green);
    auto mirrorMat = naiveMetalShader.CreateHostMaterial(metal);
    auto dispersionMat = dispersionShader.CreateHostMaterial(metal);

    NaiveDielectricsData dielectrics;
    dielectrics.refractivity = 1.5f;

    auto dielectricsMat = naiveDielectrixShader.CreateHostMaterial(dielectrics);


    DisneyPbrData pbrData;
    pbrData.baseColor = make_float3(0.2f, 0.8f, 0.2f);
    pbrData.roughness = 0.5f;
    pbrData.specular = 0.2f;
    pbrData.subsurface = 1.0f;
    pbrData.specularTint = 0.6f;
    pbrData.metallic = 0.5f;

    auto roughPbr = disneyPbrShader.CreateHostMaterial(pbrData);

    Mesh plane, cube, leftPlane, rightPlane, light, cube2, prism;

    // plane.AddCube(make_float3(0, 0, 3), make_float3(3, 3, 0.1f));
    plane.AddCube(make_float3(0, -3, 0), make_float3(6, 0.1f, 6));
    plane.AddCube(make_float3(0, 3, 0), make_float3(3, 0.1f, 3));
    plane.material = whiteMat;

    leftPlane.AddCube(make_float3(-3, 0, 0), make_float3(0.1f, 3, 3));
    leftPlane.material = redMat;

    rightPlane.AddCube(make_float3(3, 0, 0), make_float3(0.1f, 3, 3));
    rightPlane.material = blueMat;

    cube2.AddCube(make_float3(0, 0, 0), make_float3(0.75f, 0.75F, 0.75f));
    cube2.Rotate(-30, make_float3(0, 1, 0));
    cube2.Move(make_float3(1, -2, -1));
    cube2.material = whiteMat;

    cube.AddCube(make_float3(0, 0, 0), make_float3(2.f, 2.f, 2.f));
    cube.Rotate(30, make_float3(0, 1, 0));
    // cube.Move(make_float3(-1, -1, 1.5f));
    cube.material = dispersionMat;

    light.AddCube(make_float3(0, 2.5, 0), make_float3(1.f, 0.2, 1.f));
    light.material = whiteLightMat;

    Sphere haha;

    haha.position = make_float3(0, 0, 2);
    haha.radius = 1;
    haha.material = roughPbr;

    Sphere test;
    test.position = make_float3(-1, 0, 1);
    test.radius = 1.5f;
    // test.material.emission = make_float3(1.f, 1.f, 1.f);
    test.material = mirrorMat;

    Curve curve;

    curve.position = make_float3(0, -1, 1);
    curve.points = { make_float3(1, -1, 0), make_float3(2, 0, 0) , make_float3(1, 1, 0) };
    curve.theta = 0;
    curve.material = mirrorMat;

    Mesh bunny = Mesh::LoadObj("../models/bunny.obj")[0];
    bunny.material = dielectricsMat;
    bunny.Scale(make_float3(1.5f));
    bunny.Move(make_float3(0.5f, -2.5, -3));

    prism.AddPrism(make_float3(0.5f, 1.0f, 0.5f), 0.5f, make_float3(2, 1.0, 0.5));
    prism.material = dielectricsMat;
    prism.Scale(make_float3(2, 2, 2));
    prism.Move(make_float3(-2, -2, -2));


    // renderer.AddCurve(std::move(curve));
    renderer.AddMesh(std::move(bunny));
    renderer.AddMesh(std::move(plane));
    // renderer.AddMesh(std::move(cube));
    // renderer.AddMesh(std::move(cube2));
    renderer.AddMesh(std::move(leftPlane));
    renderer.AddMesh(std::move(rightPlane));
    renderer.AddMesh(std::move(light));

    renderer.AddSphere(std::move(haha));
    // renderer.AddSphere(std::move(test));
    // renderer.AddMesh(std::move(prism));

    PathTracerWindow window(&renderer);

    window.Run();
}


extern "C" int main(int ac, char** av) {
    
    try {
        InitCudaAndOptix();
        Scene2();
    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }


    return 0;
}