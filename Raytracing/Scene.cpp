#include "Scene.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "util.h"
#include <algorithm>
using namespace std;
Scene::Scene()
{
	memset(&data,0, sizeof(data));
	Plane p;
	planes.push_back(p);
	Lambert l;
	l.color = Color(0.9f,0.9f,0.9f);
	lambertShaders.push_back(l);
	l.color = (Color(0.1f,0.7f,0.3f));
	lambertShaders.push_back(l);
	Phong phong;
	phong.color = Color(0.3f, 0.9f, 0.9f);
	phong.exponent = 40;
	phongShaders.push_back(phong);
	Sphere s(Vector(10.0f, -10.0f, 30.0f),10);
	spheres.push_back(s);
	Cube c(Vector( -10.0f, -10.0f, 30.0f),10);
	cubes.push_back(c);
	Node n;
	n.geometry = 0;
	n.shader = 0;
	n.geometryType = Geometries::geometryPlane;
	n.shaderType = Shaders::shaderLambert;
	nodes.push_back(n);
	Node n2;
	n2.geometry = 0;
	n2.shader = 0;
	n2.geometryType = Geometries::geometrySphere;
	n2.shaderType = shaderPhong;
	nodes.push_back(n2);
	Node n3;
	n3.geometry = 0;
	n3.shader = 1;
	n3.geometryType = Geometries::geometryCube;
	n3.shaderType = shaderLambert;
	nodes.push_back(n3);
	PointLight light;
	light.col = Color(0.3f, 1.0f, 1.0f);
	light.pos = Vector(0.0f, -30.0f,0.0f);
	light.power = 500.0f;
	pointLights.push_back(light);
	light.col = Color(1.0f, 0.0f, 0.0f);
	light.pos = Vector(20.0f, -30.0f,0.0f);
	light.power = 500.0f;
	pointLights.push_back(light);
	light.col = Color(1.0f, 1.0f, 1.0f);
	light.pos = Vector(0.0f, -30.0f, 60.0f)*2;	
	pointLights.push_back(light);
	loadToDevice();
}
void Scene::cleanDevice()
{
	HANDLE_ERROR(cudaFree(data.pointLightsDev));
	HANDLE_ERROR(cudaFree(data.planesDev));
	HANDLE_ERROR(cudaFree(data.cubesDev));
	HANDLE_ERROR(cudaFree(data.spheresDev));
	HANDLE_ERROR(cudaFree(data.lambertShadersDev));
	HANDLE_ERROR(cudaFree(data.nodes));
	HANDLE_ERROR(cudaFree(data.phongShadersDev));
}


void Scene::loadToDevice()
{
	cleanDevice();
	data.pointLightsCount = getPointLightsCount();
	data.planesCount = getPlanesCount();
	data.cubesCount = getCubesCount();
	data.spheresCount = getSpheresCount();
	data.lambertShadersCount =getLambertShadersCount();
	data.nodesCount = getNodesCount();
	data.phongShadersCount = getPhongShadersCount();

	HANDLE_ERROR(cudaMalloc((void**)&data.pointLightsDev, data.pointLightsCount *sizeof(PointLight)));
	HANDLE_ERROR(cudaMalloc((void**)&data.planesDev, data.planesCount * sizeof(Plane)));
	HANDLE_ERROR(cudaMalloc((void**)&data.cubesDev, data.cubesCount * sizeof(Cube)));
	HANDLE_ERROR(cudaMalloc((void**)&data.spheresDev, data.spheresCount * sizeof(Sphere)));
	HANDLE_ERROR(cudaMalloc((void**)&data.lambertShadersDev, data.lambertShadersCount * sizeof(Lambert)));
	HANDLE_ERROR(cudaMalloc((void**)&data.nodes, data.nodesCount*sizeof(Node)));
	HANDLE_ERROR(cudaMalloc((void**)&data.phongShadersDev, data.phongShadersCount * sizeof(Phong)));

	HANDLE_ERROR(cudaMemcpy((void*)data.pointLightsDev, (void*)pointLights.data(), sizeof(PointLight)* data.pointLightsCount, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void*)data.planesDev, (void*)planes.data(), sizeof(Plane)* data.planesCount, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(data.cubesDev, cubes.data(), sizeof(Cube)* data.cubesCount, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(data.spheresDev, spheres.data(), sizeof(Sphere)* data.spheresCount, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(data.lambertShadersDev, lambertShaders.data(), sizeof(Lambert)* data.lambertShadersCount, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(data.nodes, nodes.data(), data.nodesCount * sizeof(Node), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(data.phongShadersDev, phongShaders.data(), data.phongShadersCount * sizeof(Phong), cudaMemcpyHostToDevice));
}