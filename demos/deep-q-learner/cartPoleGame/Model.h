// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef MODEL_H
#define MODEL_H

#define CART_MASS 10.0
#define POLE_MASS 0.1
#define POLE_LENGTH 10.0 // in pixels
#define GRAVITATIONAL_CONSTANT 1.0
#define CART_FRICTION 0.1 // resistance to cart motion
#define POLE_FRICTION 0.3 // resistance to pole angular motion
#define BOUNDARY 1.0 // how far from the center the cart is allowed to travel

# include "../../../src/vec.h"


class Model
{
public:
	double cart_position; // 0 = center
	double cart_velocity; // in position units per time frame
	double pole_angle; // 0 is straight down.
	double pole_angular_velocity; // in radians per time frame
	double applied_force; // force applied to the cart by the agent
    double dt;

	// Constructor
	Model();

	// Destructor
	~Model();

	// Restarts the game
	void reset();

	// moves cart and pole system to next state in time
	void update();

	// Applies a force to the cart
	void applyForce(double f)
	{
		applied_force = f;
	}
};

#endif
