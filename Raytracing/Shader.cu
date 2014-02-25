#include "Shader.cuh"
#include "Scene.h"
extern bool __device__ checkVisibility(Vector firstPoint, Vector secondPoint, SceneDeviceData sceneData );
extern Color __device__ raytrace(const Ray& ray, SceneDeviceData sceneData);
Color __device__ Lambert::eval(const Ray& ray, IntersectionData& data, SceneDeviceData sceneData)
{
	Color res = Color(0.0f, 0.0f, 0.0f);
	Vector n = faceforward(ray.dir, data.normal);
	for(int i = 0; i < sceneData.pointLightsCount ; ++i)
	{
		PointLight& l = sceneData.pointLightsDev[i];
		if(!checkVisibility(data.p + 1e-4*n, l.pos, sceneData))
			continue;
		Vector lightDir = l.pos - data.p;
		lightDir.normalize();
		float cosTheta = lightDir * n;
		if(cosTheta > 0)
			res += (l.col * l.power) / (data.p - l.pos).lengthSqr() * cosTheta;
	}
	return res* color;
}
Color __device__ Phong::eval(const Ray& ray, IntersectionData& data,const  SceneDeviceData& sceneData)
{
	Color diffuse = Color(0.0f, 0.0f, 0.0f);
	Color specular = diffuse;
	Vector n = faceforward(ray.dir, data.normal);
	for(int i = 0; i < sceneData.pointLightsCount ; ++i)
	{
		PointLight& l = sceneData.pointLightsDev[i];
		if(!checkVisibility(data.p + 1e-4*n, l.pos, sceneData))
			continue;
		Vector lightDir = l.pos - data.p;
		lightDir.normalize();
		float cosTheta = lightDir * n;
		Color baseLight = (l.col * l.power) / (data.p - l.pos).lengthSqr() * cosTheta;
		if(cosTheta > 0)
			diffuse	 += baseLight * cosTheta;
		float cosGamma = reflect(-lightDir,n)*(-ray.dir);
		if(cosGamma >0)
			specular += baseLight*powf(cosGamma, exponent); 
	}
	return color*diffuse + specular;
	//return raytrace(newRay, sceneData);
}