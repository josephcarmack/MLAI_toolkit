// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef VIEW_H
#define VIEW_H

#include <SDL2/SDL.h>


class Model;

#define GROUND_Y 400
#define WINDOW_WIDTH 500
#define CART_WIDTH 100
#define CART_HEIGHT 30
#define CART_WHEEL_DIAMETER 10
#define SCALE 100.0


class View
{
public:
	SDL_Window* m_pWindow;
	SDL_Renderer* m_pRenderer;
	SDL_Surface* m_pPrimarySurface;

	Model& model;
	volatile bool visualizing;


	// Constructor
	View(Model& m, int w, int h);

	// Destructor
	virtual ~View();

	// Draws the screen
	void update();

protected:
	void drawRect(int x, int y, int w, int h);
};



#endif
