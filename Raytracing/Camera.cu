#include "camera.h"
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