// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <vector>
#include <unistd.h>
#include "cartPoleGame/Model.h"
#include "cartPoleGame/View.h"
#include "cartPoleGame/Controller.h"

using std::cerr;


void mili_sleep(unsigned int nMiliseconds)
{
#ifdef WINDOWS
	MSG aMsg;
	while(PeekMessage(&aMsg, NULL, WM_NULL, WM_NULL, PM_REMOVE))
	{
		TranslateMessage(&aMsg);
		DispatchMessage(&aMsg);
	}
	SleepEx(nMiliseconds, 1);
#else
	nMiliseconds ? usleep(nMiliseconds * 1024) : sched_yield();
#endif
}



int main(int argc, char *argv[])
{
	int nRet = 0;
	try
	{
		bool keepRunning = true;
		Model m;
		View v(m, 500, 500);
		Controller c(m, v, &keepRunning);
		while(keepRunning)
		{
			if(v.visualizing)
				v.update();
			mili_sleep(30);
			c.update();
		}
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

