#pragma once

#include "sutil\Trackball.h"
#include "sutil\Camera.h"
#include <GLFW/glfw3.h>
#include "renderer.h"
#include <sutil\sutil.h>
#include <sutil\GLDisplay.h>

class PathTracerWindow {

public:
    PathTracerWindow(PathTracer* renderer) {
        this->renderer = renderer;
    }

    void Run() {
        window = sutil::initUI("Positron Sniper Rifle", renderer->GetScreenSize().x, renderer->GetScreenSize().y);

        glfwSetWindowUserPointer(window, this);
        glfwSetMouseButtonCallback(window, OnMouseButton);
        glfwSetCursorPosCallback(window, OnMouseMove);
        glfwSetWindowSizeCallback(window, OnWindowResize);
        glfwSetWindowIconifyCallback(window, OnWindowIconify);
        glfwSetScrollCallback(window, OnScroll);

        windowCamera.setEye(make_float3(0, 0, -20.f));
        windowCamera.setLookat(make_float3(0, 0, 0));
        windowCamera.setUp(make_float3(0.0f, 1.0f, 0.0f));
        windowCamera.setFovY(35.0f);

        trackball.setCamera(&windowCamera);
        trackball.setMoveSpeed(0.1f);
        trackball.setReferenceFrame(
            make_float3(-1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 1.0f, 0.0f)
        );
        trackball.setGimbalLock(true);

        renderer->Init();

        renderer->Resize(make_int2(512, 512));

        sutil::GLDisplay glDisplay(sutil::BufferImageFormat::FLOAT3);

        SetCamera();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            renderer->Render();

            int framebufSizeX = 0;  
            int framebufSizeY = 0;  
            glfwGetFramebufferSize(window, &framebufSizeX, &framebufSizeY);

            glDisplay.display(renderer->GetScreenSize().x, renderer->GetScreenSize().y, framebufSizeX, framebufSizeY, renderer->outputBuffer->getPBO());

            glfwSwapBuffers(window);
        }
    }

protected:
    GLFWwindow* window;
    PathTracer* renderer;

    sutil::Trackball trackball;
    sutil::Camera windowCamera;

    int mouseButton = -1;

    bool windowMinimized = false;

	static void OnMouseButton(GLFWwindow* window, int button, int action, int mods) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        PathTracerWindow& wthis = *(PathTracerWindow*)glfwGetWindowUserPointer(window);

        if (action == GLFW_PRESS) {
            wthis.mouseButton = button;
            wthis.trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
        } else {
            wthis.mouseButton = -1;
        }
	}

    static void OnMouseMove(GLFWwindow* window, double xpos, double ypos) {
        PathTracerWindow& wthis = *(PathTracerWindow*)glfwGetWindowUserPointer(window);
        sutil::Trackball& trackball = wthis.trackball;

        int2 size = wthis.renderer->GetScreenSize();

        if (wthis.mouseButton == GLFW_MOUSE_BUTTON_LEFT) {
            trackball.setViewMode(sutil::Trackball::LookAtFixed);
            trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), size.x, size.y);
            wthis.SetCamera();
        } else if (wthis.mouseButton == GLFW_MOUSE_BUTTON_RIGHT) {
            trackball.setViewMode(sutil::Trackball::EyeFixed);
            trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), size.x, size.y);
            wthis.SetCamera();
        }
    }

    static void OnWindowResize(GLFWwindow* window, int32_t res_x, int32_t res_y) {
        PathTracerWindow& wthis = *(PathTracerWindow*)glfwGetWindowUserPointer(window);

        if (wthis.windowMinimized)
            return;

        sutil::ensureMinimumSize(res_x, res_y);

        wthis.renderer->Resize(make_int2(res_x, res_y));
    }

    static void OnWindowIconify(GLFWwindow* window, int32_t iconified) {
        PathTracerWindow& wthis = *(PathTracerWindow*)glfwGetWindowUserPointer(window);

        wthis.windowMinimized = (iconified > 0);
    }

    static void OnScroll(GLFWwindow* window, double xscroll, double yscroll) {
        PathTracerWindow& wthis = *(PathTracerWindow*)glfwGetWindowUserPointer(window);
        if (wthis.trackball.wheelEvent((int)yscroll)) {
            wthis.SetCamera();
        }
    }

    void SetCamera() {

        renderer->SetCamera(PathTracerCameraSetting{ windowCamera.eye(), windowCamera.lookat(), windowCamera.up(), windowCamera.fovY()});
    }


	
};