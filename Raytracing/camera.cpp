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

#include "camera.h"
#include "matrix.h"
#include "util.h"
#include "cuda.h"

void Camera::beginFrame(int frameWidth, int frameHeight)
{
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
	double x = -aspect;
	double y = +1;
	
	Vector corner = Vector(x, y, 1);
	Vector center = Vector(0, 0, 1);
	
	double lenXY = (corner - center).length();
	double wantedLength = tan(toRadians(fov / 2));
	
	double scaling = wantedLength / lenXY;
	
	x *= scaling;
	y *= scaling;
	
	
	this->upLeft = Vector(x, y, 1);
	this->upRight = Vector(-x, y, 1);
	this->downLeft = Vector(x, -y, 1);
	
	Matrix rotation = rotationAroundZ(toRadians(roll))
	                * rotationAroundX(toRadians(pitch))
	                * rotationAroundY(toRadians(yaw));
	upLeft *= rotation;
	upRight *= rotation;
	downLeft *= rotation;
	rightDir = Vector(1, 0, 0) * rotation;
	upDir    = Vector(0, 1, 0) * rotation;
	frontDir = Vector(0, 0, 1) * rotation;
	
	upLeft += pos;
	upRight += pos;
	downLeft += pos;
}



void Camera::move(double dx, double dz)
{
	pos += dx * rightDir;
	pos += dz * frontDir;
}

void Camera::rotate(double dx, double dz)
{
	pitch += dz;
	if (pitch >  90) pitch = 90;
	if (pitch < -90) pitch = -90;
	
	yaw += dx;
}
