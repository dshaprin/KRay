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
#include <gl/glew.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cmath>
#include "camera.h"
#include <iostream>
#include "init.h"

using std::cout;

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;
Ray __device__ Camera::getScreenRay(double x, double y, int camera)
{
	Ray result; // A, B -     C = A + (B - A) * x
	result.start = pos;
	Vector target = upLeft + 
		(upRight - upLeft) * (x / (double) frameWidth) +
		(downLeft - upLeft) * (y / (double) frameHeight);
	
	// A - camera; B = target
	result.dir = target - this->pos;
	
	result.dir.normalize();
	
	if (camera != CAMERA_CENTER) {
		// offset left/right for stereoscopic rendering
		result.start += rightDir * (camera == CAMERA_RIGHT ? +stereoSeparation : -stereoSeparation);
	}
	
	/*if (!dof) return result;
	
	double cosTheta = dot(result.dir, frontDir);
	double M = focalPlaneDist / cosTheta;
	
	Vector T = result.start + result.dir * M;
	
	//Random& R = getRandomGen();
	double dx, dy;
	R.unitDiscSample(dx, dy);
	
	dx *= discMultiplier;
	dy *= discMultiplier;
	
	result.start = this->pos + dx * rightDir + dy * upDir;
	if (camera != CAMERA_CENTER) {
		result.start += rightDir * (camera == CAMERA_RIGHT ? +stereoSeparation : -stereoSeparation);
	}
	result.dir = (T - result.start);
	result.dir.normalize();*/
	return result;
}
__constant__ Camera cameraDev[1];
__global__ void kernel( uchar4 *ptr, int width, int height ) 
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(y < height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int offset = x + y * blockDim.x * gridDim.x;
		Ray ray = cameraDev->getScreenRay(x, y);
		ptr[offset].x = ptr[offset].y = ptr[offset].z = 0;
		if(fabsf(ray.dir.x) > fabsf(ray.dir.y) && fabsf(ray.dir.x) > fabsf(ray.dir.z))
			if(ray.dir.x > 0)
				ptr[offset].x = 255;
			else
				ptr[offset].y = 255;
		else if(fabsf(ray.dir.y) > fabsf(ray.dir.z))
			if(ray.dir.y > 0)
				ptr[offset].z =  255;
			else{
				ptr[offset].x = 255;
				ptr[offset].y = 255;
			}
		else if(ray.dir.z > 0){
			ptr[offset].x = 255;
			ptr[offset].z = 255;
		}
		else{
			ptr[offset].y = 255;
			ptr[offset].z = 255;
		}
		ptr[offset].w = 255;

	}
	
}
void render(uchar4* devPtr, cudaGraphicsResource *resource, Camera& camera, int width, int height)
{  
	HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
    size_t  size;
    HANDLE_ERROR( 
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
                                              &size, 
                                              resource) );
	camera.beginFrame(width, height);
	HANDLE_ERROR(cudaMemcpyToSymbol(cameraDev, &camera, sizeof(Camera)));
	dim3    grids(width/16,height/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( devPtr, width, height );
	glutPostRedisplay();
	HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
}
int main( int argc, char **argv )
{
	init(argc,argv);
      
    // set up GLUT and kick off main loop
    glutKeyboardFunc( key_func );
    glutDisplayFunc( draw_func );
    glutMainLoop();
	
}
