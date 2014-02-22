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
extern int windowWidth, windowHeight;

extern Scene* scene;
extern GLuint  bufferObj;
extern cudaGraphicsResource *resource;
extern uchar4* devPtr;
extern Camera camera;

extern std::vector<Plane*> planes;
extern Plane* planesDev;
void keyFunc( unsigned char key, int x, int y );
void mouseFunc(int x, int y);
void update();
void drawFunc( void );
void initCamera();
void init(int argc, char* argv[]);
#endif INIT_H
