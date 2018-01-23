// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "View.h"
#include "Model.h"
#include "../../../src/error.h"
#include <SDL2/SDL_image.h>


View::View(Model& m, int w, int h)
: model(m), visualizing(true)
{
	// Init SDL stuff
	if(SDL_Init(SDL_INIT_VIDEO) < 0)
		throw Ex("Unable to Init SDL: ", SDL_GetError());
	if(!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1"))
		throw Ex("Unable to Init hinting: ", SDL_GetError());
	m_pWindow = SDL_CreateWindow("Pole Cart", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, SDL_WINDOW_SHOWN);
	if(m_pWindow == NULL)
		throw Ex("Unable to create SDL Window: ", SDL_GetError());
	m_pPrimarySurface = SDL_GetWindowSurface(m_pWindow);
	m_pRenderer = SDL_GetRenderer(m_pWindow);
    	if(m_pRenderer == NULL)
	m_pRenderer = SDL_GetRenderer(m_pWindow);
	if(m_pRenderer == NULL)
		m_pRenderer = SDL_CreateRenderer(m_pWindow, -1, SDL_RENDERER_ACCELERATED);
	if(m_pRenderer == NULL)
		throw Ex("Unable to create renderer");

	// Initialize image loading for PNGs
	if(!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
		throw Ex("Unable to init SDL_image: ", IMG_GetError());
	}
}

// virtual
View::~View()
{
	if(m_pRenderer)
		SDL_DestroyRenderer(m_pRenderer);
//	if(m_pWindow)
//		SDL_DestroyWindow(m_pWindow);
	IMG_Quit();
//	SDL_Quit();
}

void View::drawRect(int x, int y, int w, int h)
{
	SDL_RenderDrawLine(m_pRenderer, x, y, x + w, y);
	SDL_RenderDrawLine(m_pRenderer, x, y + h, x + w, y + h);
	SDL_RenderDrawLine(m_pRenderer, x, y, x, y + h);
	SDL_RenderDrawLine(m_pRenderer, x + w, y, x + w, y + h);
}

void View::update()
{
	// Clear the screen
	SDL_SetRenderDrawColor(m_pRenderer, 0xff, 0xff, 0xff, 0xff);
	SDL_RenderClear(m_pRenderer);
	
	// Draw the ground and boundaries
	SDL_SetRenderDrawColor(m_pRenderer, 0x00, 0x00, 0x00, 0xff);
	int left = WINDOW_WIDTH / 2 - (int)(BOUNDARY * SCALE) - CART_WIDTH / 2;
	int right = WINDOW_WIDTH / 2 + (int)(BOUNDARY * SCALE) + CART_WIDTH / 2;
	SDL_RenderDrawLine(m_pRenderer, left, GROUND_Y, right, GROUND_Y);
	SDL_RenderDrawLine(m_pRenderer, left, GROUND_Y, left, GROUND_Y - 30);
	SDL_RenderDrawLine(m_pRenderer, right, GROUND_Y, right, GROUND_Y - 30);

	// Draw the cart
	int pole_x = WINDOW_WIDTH / 2 + (int)(SCALE * model.cart_position);
	int cart_x = pole_x - CART_WIDTH / 2;
	int cart_y = GROUND_Y - CART_WHEEL_DIAMETER - CART_HEIGHT;
	drawRect(cart_x, cart_y, CART_WIDTH, CART_HEIGHT);
	//SDL_RenderDrawOval(m_pRenderer, cart_x, cart_y + CART_HEIGHT, CART_WHEEL_DIAMETER, CART_WHEEL_DIAMETER);
	//SDL_RenderDrawOval(m_pRenderer, cart_x + CART_WIDTH - CART_WHEEL_DIAMETER, cart_y + cart_height, CART_WHEEL_DIAMETER, CART_WHEEL_DIAMETER);

	// Draw the pole
	double horiz_reach = SCALE * cos(model.pole_angle + M_PI / 2.0);
	double vert_reach = SCALE * sin(model.pole_angle + M_PI / 2.0);
	SDL_RenderDrawLine(m_pRenderer, pole_x, cart_y, pole_x + (int)horiz_reach, cart_y + (int)vert_reach);
	
	SDL_RenderPresent(m_pRenderer);
}

