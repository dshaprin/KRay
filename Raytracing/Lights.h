#pragma once
#ifndef LIGHTS_COUNT
#define LIGHTS_COUNT
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "vector.h"
#include "color.h"
#include <vector>
struct PointLight
{
	Color col;
	float power;
	Vector pos;
};
#endif