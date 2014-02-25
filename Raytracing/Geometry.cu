#include "geometry.h"
bool __device__ Plane::intersect(const Ray& ray, IntersectionData& data)
{
	// intersect a ray with a XZ plane:
	// if the ray is pointing to the horizon, or "up", but the plane is below us,
	// of if the ray is pointing down, and the plane is above us, we have no intersection
	if ((ray.start.y > y && ray.dir.y > -1e-9) || (ray.start.y < y && ray.dir.y < 1e-9))
		return false;
	else {
		float yDiff = ray.dir.y;
		float wantYDiff = ray.start.y - this->y;
		float mult = wantYDiff / -yDiff;
		
		// if the distance to the intersection (mult) doesn't optimize our current distance, bail out:
		if (mult > data.dist) return false;
		
		Vector p = ray.start + ray.dir * mult;
		if (fabs(p.x) > limit || fabs(p.z) > limit) 
			return false;
		
		// calculate intersection:
		data.p = p;
		data.dist = mult;
		data.normal = Vector(0, 1, 0);
		data.dNdx = Vector(1, 0, 0);
		data.dNdy = Vector(0, 0, 1);
		data.u = data.p.x;
		data.v = data.p.z;
//		data.g = this;
		return true;
	}
}
bool __device__ Sphere::intersect(const Ray& ray, IntersectionData& info)
{
	// compute the sphere intersection using a quadratic equation:
	Vector H = ray.start - center;
	float A = ray.dir.lengthSqr();
	float B = 2 * dot(H, ray.dir);
	float C = H.lengthSqr() - R*R;
	float Dscr = B*B - 4*A*C;
	if (Dscr < 0) return false; // no solutions to the quadratic equation - then we don't have an intersection.
	float x1, x2;
	x1 = (-B + sqrt(Dscr)) / (2*A);
	x2 = (-B - sqrt(Dscr)) / (2*A);
	float sol = x2; // get the closer of the two solutions...
	if (sol < 0) sol = x1; // ... but if it's behind us, opt for the other one
	if (sol < 0) return false; // ... still behind? Then the whole sphere is behind us - no intersection.
	
	// if the distance to the intersection doesn't optimize our current distance, bail out:
	if (sol > info.dist) return false;
	
	info.dist = sol;
	info.p = ray.start + ray.dir * sol;
	info.normal = info.p - center; // generate the normal by getting the direction from the center to the ip
	info.normal.normalize();
	float angle = atan2(info.p.z - center.z, info.p.x - center.x);
	info.u = (PI + angle)/(2*PI);
	info.v = 1.0 - (PI/2 + asin((info.p.y - center.y)/R)) / PI;
	info.dNdx = Vector(cos(angle + PI/2), 0, sin(angle + PI/2));
	info.dNdy = info.dNdx ^ info.normal;
	return true;
}

inline __device__  bool Cube::intersectCubeSide(const Ray& ray, const Vector& center, IntersectionData& data)
{
	if (fabs(ray.dir.y) < 1e-9) return false;

	float halfSide = this->side * 0.5;
	bool found = false;
	for (int side = -1; side <= 1; side += 2) {
		float yDiff = ray.dir.y;
		float wantYDiff = ray.start.y - (center.y + side * halfSide);
		float mult = wantYDiff / -yDiff;
		if (mult < 0) continue;
		if (mult > data.dist) continue;
		Vector p = ray.start + ray.dir * mult;
		if (p.x < center.x - halfSide ||
			p.x > center.x + halfSide ||
			p.z < center.z - halfSide ||
			p.z > center.z + halfSide) continue;
		data.p = ray.start + ray.dir * mult;
		data.dist = mult;
		data.normal = Vector(0, side, 0);
		data.dNdx = Vector(1, 0, 0);
		data.dNdy = Vector(0, 0, side);
		data.u = data.p.x - center.x;
		data.v = data.p.z - center.z;
		found = true;	
	}
	return found;
}

bool __device__  Cube::intersect(const Ray& ray, IntersectionData& data)
{
	// check for intersection with the negative Y and positive Y sides
	bool found = intersectCubeSide(ray, center, data);
	
	// check for intersection with the negative X and positive X sides
	if (intersectCubeSide(project(ray, 1, 0, 2), project(center, 1, 0, 2), data)) {
		found = true;
		data.normal = unproject(data.normal, 1, 0, 2);
		data.p = unproject(data.p, 1, 0, 2);
	}

	// check for intersection with the negative Z and positive Z sides
	if (intersectCubeSide(project(ray, 0, 2, 1), project(center, 0, 2, 1), data)) {
		found = true;
		data.normal = unproject(data.normal, 0, 2, 1);
		data.p = unproject(data.p, 0, 2, 1);
	}
	return found;
}