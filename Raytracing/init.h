#pragma once
#ifndef INIT_H
#define INIT_H
#include <Windows.h>
#include <stdio.h>

#include <GL\glew.h>
#include <gl_helper.h>
#include <GL\GL.h>
#include "util.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cmath>
#include "camera.h"
#include <iostream>
#include <algorithm>
#include "geometry.h"
#include "kernel.cuh"
#include "Scene.h"

using std::cout;
const char* const TITLE = "Raytracer";
const int WIDTH = 800, HEIGHT = 592;

extern Scene* scene;
extern GLuint  bufferObj;
extern cudaGraphicsResource *resource;
extern uchar4* devPtr;
extern Camera camera;

extern std::vector<Plane*> planes;
extern Plane* planesDev;
void key_func( unsigned char key, int x, int y );
void update();
void draw_func( void );
void initCamera();
void init(int argc, char* argv[]);
#endif INIT_H
