#pragma once
#ifndef SHADER_CUH
#define SHADER_CUH
#include "cuda.h"
#include "color.h"
#include "vector.h"
#include "geometry.h"
#include "Lights.h"
//#include "Scene.h"
class SceneDeviceData;
class Lambert
{
public:
	Color color;
	Color __device__ eval(const Ray& ray, IntersectionData& data, SceneDeviceData sceneData);
};
class Phong
{
	
public:
	Color color;
	float exponent;
	Color __device__ eval(const Ray& ray, IntersectionData& data,const SceneDeviceData& sceneData);
			
};
#endif SHADER_CUH