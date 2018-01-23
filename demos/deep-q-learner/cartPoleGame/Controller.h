// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "Model.h"
#include "View.h"
#include <SDL2/SDL.h>
#include "../../../src/rand.h"
#include "../../../src/neuralnet.h"


class Controller
{
public:
	Model& m_model;
	View& m_view;
	bool* m_pKeepRunning;
	bool m_spaceDown;
	bool m_training;
	bool m_greedy;
	bool m_replayOut;
    Vec m_state;
    Rand m_rand;

    // stuff for q-learning
    NeuralNet* m_Q;
    size_t bufSize;
    size_t stateSize;
    size_t iter;
    size_t exSt;
    Matrix m_replayQue;
    Matrix m_qFactors;
    int action;
    int numActions;
    double qFact_a;
    double qFact_c;
    double epsilon;
    double alphaK;
    double gamma;
    double learningRate;
    Vec si;
    Vec sj;
    Vec qIn;
    long unsigned int tim;
    long unsigned int life;

public:
	// Constructor
	Controller(Model& m, View& v, bool* pKeepRunning);

	// Destructor
	virtual ~Controller();

	// Responds to keyboard and mouse state
	void update();

    // get models state
    void getState(Model& mod,Vec& state);

    // get models state
    void buildQin(Vec& state,int act, Vec& out);

    // adds new pattern to replay buffer
    void addToQueue(int it,Vec& qin,double qf);
};


#endif
