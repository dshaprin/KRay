/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cmath>
#include "camera.h"
#include <iostream>
#include "kernel.cuh"

using std::cout;
const char* const TITLE = "Raytracer";
const int WIDTH = 800, HEIGHT = 592;


GLuint  bufferObj;
cudaGraphicsResource *resource;
uchar4* devPtr;
Camera camera;

static void key_func( unsigned char key, int x, int y ) {
    switch (key) {
		case 's':
			camera.rotate(0,1);
		break;
		case 'w':
			camera.rotate(0,-1);
		break;
		case 'a':
			camera.rotate(1,0);
		break;
		case 'd':
			camera.rotate(-1,0);
		break;
        case 27:
            // clean up OpenGL and CUDA

            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
			
            exit(0);
    }
}
void update()
{
	  // do work with the memory dst being on the GPU, gotten via mapping
	render(devPtr,resource, camera, WIDTH, HEIGHT);
}
static void draw_func( void ) 
{
	update();
	glDrawPixels( WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glutSwapBuffers();
}
void initCamera()
{
	camera.init();
	camera.pos = Vector(0.0f, 10.0f, 0.0f);
	camera.yaw = -90.0;
	camera.pitch = 0.0;
	camera.roll = 0.0;
	camera.fov = 60;

}
void init(int argc, char* argv[])
{
	initCamera();
	cudaDeviceProp  prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( WIDTH, HEIGHT);
    glutCreateWindow( TITLE );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // the first three are standard OpenGL, the 4th is the CUDA reg 
    // of the bitmap these calls exist starting in OpenGL 1.5
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT* 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

    HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );

  
}

