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
#include "util.h"


#include "cuda.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cmath>
#include "camera.h"
#include <iostream>
#include "init.h"
#include "geometry.h"
#include "FrameBuffer.h"
using std::cout;

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;


__constant__ Camera cameraDev[1];
bool __device__ checkVisibility(Vector firstPoint, Vector secondPoint, SceneDeviceData sceneData )
{
	Vector dir = secondPoint - firstPoint;
	float distance = dir.length();
	dir /= distance;
	Ray ray(firstPoint,dir);
	ray.start = firstPoint;
	ray.dir = dir;
	ray.flags |= RF_SHADOW;
	
	for(int i = 0; i < sceneData.nodesCount; ++i){
		Node& cur = sceneData.nodes[i];
		IntersectionData intersectionData;
		intersectionData.dist = distance;
		switch(cur.geometryType)
		{
			case geometrySphere:
				if(sceneData.spheresDev[cur.geometry].intersect(ray, intersectionData))
						return false;
				break;
			case geometryPlane:	
				if(sceneData.planesDev[cur.geometry].intersect(ray, intersectionData))
						return false;
				break;		
			case geometryCube:
				if(sceneData.cubesDev[cur.geometry].intersect(ray, intersectionData))
						return false;
				break;
			
			
		}
	}
	return true;
}
Color __device__ raytrace(const Ray& ray, SceneDeviceData sceneData)
{
	Node * n = nullptr;
	IntersectionData intersectionData;
	intersectionData.dist = INF;
	for(int i = 0; i < sceneData.nodesCount; ++i){
		Node& cur = sceneData.nodes[i];
		switch(cur.geometryType)
		{
			case geometryPlane:
				
				if(sceneData.planesDev[cur.geometry].intersect(ray, intersectionData))
					n = &cur;
				break;
				
			case geometryCube:
				if(sceneData.cubesDev[cur.geometry].intersect(ray, intersectionData))
					n = &cur;
				break;
			case geometrySphere:
				if(sceneData.spheresDev[cur.geometry].intersect(ray, intersectionData))
					n = &cur;
				break;
			
			default:
				return Color(1,0,1);
		}
		
		
	}
	if(n != nullptr)
		switch(n->shaderType)
		{
			case shaderLambert:
				return sceneData.lambertShadersDev[n->shader].eval(ray, intersectionData, sceneData);
		}

	return Color(0.0f, 0.0f, 1.0f);
}
__global__ void kernel( uchar4 *ptr, int width, int height, SceneDeviceData data) 
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(y < height && x < width)
	{	
		int offset = x + y * blockDim.x * gridDim.x;
		Ray ray = cameraDev->getScreenRay(x, y);
		Color c = raytrace(ray, data);
		unsigned  col = c.toRGB32();
		ptr[offset].x = (( col & 0x00FF0000)>>16);
		ptr[offset].y = (( col & 0x0000FF00)>>8);		
		ptr[offset].z = ( col & 0x000000FF);
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
	dim3    grids((width + 15)/16,(height + 15)/16);
    dim3    threads(16,16);
	kernel<<<grids,threads>>>( devPtr, width, height, scene->getDeviceData());
	glutPostRedisplay();
	HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
}

int main( int argc, char **argv )
{
	init(argc,argv);
	 
    // set up GLUT and kick off main loop
    glutKeyboardFunc( keyFunc );
    glutDisplayFunc( drawFunc );
    glutMainLoop();
	
}
