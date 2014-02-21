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
#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "vector.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "random_generator.h"

enum {
	CAMERA_CENTER,
	CAMERA_LEFT,
	CAMERA_RIGHT,
};

class Camera {
	// these internal vectors describe three of the ends of the imaginary
	// ray shooting screen
	Vector upLeft, upRight, downLeft;
	Vector frontDir, rightDir, upDir;
	int frameHeight;
	int frameWidth;
public:
	Vector pos; //!< position of the camera in 3D.
	double yaw; //!< Yaw angle in degrees (rot. around the Y axis, meaningful values: [0..360])
	double pitch; //!< Pitch angle in degrees (rot. around the X axis, meaningful values: [-90..90])
	double roll; //!< Roll angle in degrees (rot. around the Z axis, meaningful values: [-180..180])
	double fov; //!< The Field of view in degrees (meaningful values: [3..160])
	double aspect; //!< The aspect ratio of the camera frame. Should usually be frameWidth/frameHeight,
	double focalPlaneDist;
	double fNumber;
	bool dof; // on or off
	int numSamples;
	double discMultiplier;
	double stereoSeparation;

	void init() {dof = false; fNumber = 1.0; focalPlaneDist = 1; numSamples = 25;
		stereoSeparation = 0;
	}
	// from SceneElement:
	void beginFrame(int frameWidth, int frameHeight); //!< must be called before each frame. Computes the corner variables, needed for getScreenRay()
	
	
	/// generates a screen ray through a pixel (x, y - screen coordinates, not necessarily integer).
	/// if the camera parameter is present - offset the rays' start to the left or to the right,
	/// for use in stereoscopic rendering
	Ray __device__ getScreenRay(double x, double y, int camera = CAMERA_CENTER);
	
	void move(double dx, double dz);
	void rotate(double dx, double dz);
};

#endif // __CAMERA_H__
