/***************************************************************************
 *   Copyright (C) 2009-2013 by Veselin Georgiev, Slavomir Kaslev et al    *
 *   admin@raytracing-bg.net                                               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

#include <math.h>
#include <ostream>
#include <iomanip>
#include "cuda_runtime_api.h"
struct Vector {
	union {
		struct { double x, y, z; };
		double components[3];
	};
	
	/////////////////////////
	__host__ __device__ Vector () {}
	__host__ __device__ Vector(double _x, double _y, double _z) { set(_x, _y, _z); }
	void __host__ __device__ set(double _x, double _y, double _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
	void __host__ __device__ makeZero(void)
	{
		x = y = z = 0.0;
	}
	inline double __host__ __device__ length(void) const
	{
		return sqrt(x * x + y * y + z * z);
	}
	inline double __host__ __device__ lengthSqr(void) const
	{
		return (x * x + y * y + z * z);
	}
	void __host__ __device__ scale(double multiplier)
	{
		x *= multiplier;
		y *= multiplier;
		z *= multiplier;
	}
	void __host__ __device__ operator *= (double multiplier)
	{
		scale(multiplier);
	}
	void __host__ __device__ operator += (const Vector& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
	}
	void __host__ __device__ operator /= (double divider)
	{
		scale(1.0 / divider);
	}
	void __host__ __device__ normalize(void)
	{
		double multiplier = 1.0 / length();
		scale(multiplier);
	}
	void __host__ __device__ setLength(double newLength)
	{
		scale(newLength / length());
	}
	
	inline __host__ __device__ double& operator[] (int index)
	{
		return components[index];
	}
	inline __host__ __device__ const  double& operator[] (int index) const
	{
		return components[index];
	}
	
	int __host__ __device__ maxDimension() const
	{
		int bi = 0;
		double maxD = fabs(x);
		if (fabs(y) > maxD) { maxD = fabs(y); bi = 1; }
		if (fabs(z) > maxD) { maxD = fabs(z); bi = 2; }
		return bi;
	}
};

inline Vector __host__ __device__ operator + (const Vector& a, const Vector& b)
{
	return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vector __host__ __device__ operator - (const Vector& a, const Vector& b)
{
	return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vector __host__ __device__ operator - (const Vector& a)
{
	return Vector(-a.x, -a.y, -a.z);
}

/// dot product
inline double __host__ __device__ operator * (const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// dot product (functional form, to make it more explicit):
inline double __host__ __device__ dot(const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// cross product
inline Vector __host__ __device__ operator ^ (const Vector& a, const Vector& b)
{
	return Vector(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

inline Vector __host__ __device__ operator * (const Vector& a, double multiplier)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}
inline Vector __host__ __device__ operator * (double multiplier, const Vector& a)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}
inline Vector __host__ __device__ operator / (const Vector& a, double divider)
{
	double multiplier = 1.0 / divider;
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

inline Vector __host__ __device__ normalize(const Vector& vec)
{
	double multiplier = 1.0 / vec.length();
	return vec * multiplier;
}

inline Vector __host__ __device__ reflect(const Vector& ray, const Vector& norm)
{
	Vector result = ray - 2 * dot(ray, norm) * norm;
	result.normalize();
	return result;
}

inline Vector __host__ __device__ faceforward(const Vector& ray, const Vector& norm)
{
	if (dot(ray, norm) < 0) return norm;
	else return -norm;
}

inline Vector __host__ __device__ project(const Vector& v, int a, int b, int c)
{
	Vector result;
	result[a] = v[0];
	result[b] = v[1];
	result[c] = v[2];
	return result;
}


inline Vector __host__ __device__ unproject(const Vector& v, int a, int b, int c)
{
	Vector result;
	result[0] = v[a];
	result[1] = v[b];
	result[2] = v[c];
	return result;
}

// flags that mark a ray in some way, so the behaviour of the raytracer can be altered.
enum RayFlags {
	// RF_DEBUG - the ray is a debug one (launched from a mouse-click on the rendered image).
	// raytrace() prints diagnostics when it encounters such a ray.
	RF_DEBUG    = 0x0001,
	
	// RF_SHADOW - the ray is a shadow ray. This hints the raytracer to skip some calculations
	// (since the IntersectionData won't be used for shading), and to disable backface culling
	// for Mesh objects.
	RF_SHADOW   = 0x0002,
	
	// RF_GLOSSY - the ray has hit some glossy surface somewhere along the way.
	// so if it meets a new glossy surface, it can safely use lower sampling settings.
	RF_GLOSSY   = 0x0004,
	
	// last constituent of a ray path was a diffuse surface
	RF_DIFFUSE  = 0x0008,
};

struct Ray {
	Vector start, dir;
	int flags;
	int depth;
	__host__ __device__ Ray() {
		flags = 0;
		depth = 0;
	}
	__host__ __device__ Ray(const Vector& _start, const Vector& _dir) {
		start = _start;
		dir = _dir;
		flags = 0;
		depth = 0;
	}
};

inline Ray __host__ __device__ project(Ray v, int a, int b, int c)
{
	v.start = project(v.start, a, b, c);
	v.dir = project(v.dir, a, b, c);
	return v;
}

// iostream Vector print routine:
inline std::ostream& operator << (std::ostream& os, const Vector& vec)
{
	os << "(" << std::fixed << std::setprecision(3) << vec.x << ", " << vec.y << ", " << vec.z << ")";
	return os;
}

/// given an unit vector a, create an orhonormed system (a, b, c). Code is deterministic.
inline void __host__ __device__ orthonormedSystem(const Vector& a, Vector& b, Vector& c)
{
	Vector temp = Vector(1, 0, 0);
	if (fabs(dot(a, temp)) > 0.99) {
		temp = Vector(0, 1, 0);
		if (fabs(dot(a, temp)) > 0.99)
			temp = Vector(0, 0, 1);
	}
	b = a ^ temp;
	b.normalize();
	c = a ^ b;
	c.normalize();
}

inline Vector __host__ __device__ refract(const Vector& i, const Vector& n, float ior)
{
	float NdotI = float(dot(i, n));
	float k = 1 - (ior * ior) * (1 - NdotI * NdotI);
	if (k < 0)
		return Vector(0, 0, 0);
	return ior * i - (ior * NdotI + sqrt(k)) * n;
}

#endif // __VECTOR3D_H__
