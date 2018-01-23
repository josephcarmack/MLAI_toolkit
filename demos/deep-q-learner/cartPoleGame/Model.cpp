// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "Model.h"
#include "../../../src/error.h"
#include "../../../src/string.h"
#include <stdlib.h>
#include <iostream>
#include <cmath>

using std::cout;


Model::Model()
{
	reset();
    dt = 1.0;
}

Model::~Model()
{
}

void Model::reset()
{
	applied_force = 0.0;
	cart_position = 0.0;
	cart_velocity = 0.0;
	pole_angle = M_PI - 0.001;
	pole_angular_velocity = 0.0;
}

void Model::update()
{
	double c = cos(pole_angle);
	double s = sin(pole_angle);

	// This was derived from the second-to-last equation at http://www.myphysicslab.com/pendulum_cart.html
	double cart_acceleration = 
		(
			POLE_MASS * POLE_LENGTH * pole_angular_velocity * pole_angular_velocity * s +
			POLE_MASS * GRAVITATIONAL_CONSTANT * s * c +
			applied_force -
			CART_FRICTION * cart_velocity +
			POLE_FRICTION / POLE_LENGTH * pole_angular_velocity * c
		) / (
			CART_MASS + POLE_MASS * s * s
		);

	// This was derived from the last equation at http://www.myphysicslab.com/pendulum_cart.html
	double pole_angular_acceleration = 
		(
			-POLE_MASS * POLE_LENGTH * pole_angular_velocity * pole_angular_velocity * s * c -
			(CART_MASS + POLE_MASS) * GRAVITATIONAL_CONSTANT * s +
			applied_force * c +
			CART_FRICTION * cart_velocity * c -
			(1.0 + CART_MASS / POLE_MASS) * (POLE_FRICTION / POLE_LENGTH) * pole_angular_velocity
		) / (
			POLE_LENGTH * (CART_MASS + POLE_MASS * s * s)
		);

	// Do kinematics
	cart_velocity += dt*cart_acceleration;
	cart_position += dt*cart_velocity;
	if(cart_position < -BOUNDARY)
	{
		cart_position = -BOUNDARY;
		cart_velocity = 0.0;
	}
	if(cart_position > BOUNDARY)
	{
		cart_position = BOUNDARY;
		cart_velocity = 0.0;
	}
	pole_angular_velocity += dt*pole_angular_acceleration;
	pole_angle += dt*pole_angular_velocity;

	// Keep the angle between -PI and PI
	if(std::abs(pole_angle) > M_PI)
		pole_angle -= (2.0 * M_PI * std::floor((pole_angle + M_PI) / (2.0 * M_PI)));

}
