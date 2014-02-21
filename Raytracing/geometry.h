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
#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <vector>
#include "vector.h"
#include "transform.h"

/// a structure, that holds info about an intersection. Filled in by Geometry::intersect() methods

struct IntersectionData {
	Vector p; //!< intersection point in the world-space
	Vector normal; //!< the normal of the geometry at the intersection point
	Vector dNdx, dNdy; 
	double dist; //!< before intersect(): the max dist to look for intersection; after intersect() - the distance found
	
	double u, v; //!< 2D UV coordinates for texturing, etc.
	
};

/// An abstract class that represents any intersectable primitive in the scene.




class Plane {
	double y; //!< y-intercept. The plane is parallel to XZ, the y-intercept is at this value
	double limit;
public:
	Plane(double _y = 0, double _limit = 1e99) { y = _y; limit = _limit; }

	bool __device__ intersect(const Ray& ray, IntersectionData& data);
	const char* getName() { return "Plane"; }
	bool isInside(const Vector& p) const { return false; }
};

class Sphere {
	Vector center;
	double R;
public:
	Sphere(const Vector& center = Vector(0, 0, 0), double R = 1): center(center), R(R) {}
	

	bool __device__ intersect(const Ray& ray, IntersectionData& data);
	const char* getName() { return "Sphere"; }
	bool isInside(const Vector& p) const { return (center - p).lengthSqr() < R*R; }
};

class Cube {
	Vector center;
	double side;
	inline bool intersectCubeSide(const Ray& ray, const Vector& center, IntersectionData& data);
public:
	Cube(const Vector& center = Vector(0, 0, 0), double side = 1): center(center), side(side) {}

	bool __device__ intersect(const Ray& ray, IntersectionData& data);	
	const char* getName() { return "Cube"; }
	bool isInside(const Vector& p) const { 
		return (fabs(p.x - center.x) <= side * 0.5 &&
				fabs(p.y - center.y) <= side * 0.5 &&
				fabs(p.z - center.z) <= side * 0.5);
	}
};



#endif // __GEOMETRY_H__
