#pragma once
#include "geometry.h"
#include "shader.cuh"
#include "lights.h"
#include <vector>
enum Lights { lightPointLight };
enum Geometries { geometryPlane, geometryCube, geometrySphere };
enum Shaders { shaderLambert, shaderPhong };
struct Node
{
	Shaders shaderType;
	Geometries geometryType;
	int geometry;
	int shader;
};
struct SceneDeviceData
{
	Node* nodes;
	int nodesCount;
	PointLight* pointLightsDev;
	int pointLightsCount;
	Plane* planesDev;
	int planesCount;
	Cube* cubesDev;
	int cubesCount;
	Sphere* spheresDev;
	int spheresCount;
	Lambert* lambertShadersDev;
	int lambertShadersCount;
	Phong *phongShadersDev;
	int phongShadersCount;
	
};
class Scene{
	std::vector< PointLight> pointLights;

	std::vector< Plane> planes;
	std::vector< Cube> cubes;
	std::vector< Sphere> spheres;

	std::vector< Lambert> lambertShaders;
	std::vector< Phong> phongShaders;
	std::vector< Node> nodes;
	//Device
	SceneDeviceData data;
	void loadToDevice();
	void cleanDevice();
	Scene(const Scene&){}
	Scene& operator= (const Scene&){return *this;}
public:
	Scene();
	size_t getPointLightsCount()const {return pointLights.size();}
	size_t getPlanesCount()const {return planes.size();}
	size_t getCubesCount()const { return cubes.size();}
	size_t getSpheresCount()const {return spheres.size();}
	size_t getLambertShadersCount()const {return lambertShaders.size();}
	size_t getNodesCount()const {return nodes.size();}
	size_t getPhongShadersCount()const {return phongShaders.size();}
	const SceneDeviceData&  getDeviceData()const
	{
		return data;
	}
	~Scene()
	{
		cleanDevice();
	}
};
