// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "Controller.h"
#include "View.h"
#include <iostream>
#include <cmath>
#include <unistd.h>
#include "../../../src/json.h"

using std::cout;

// This is the constructor
Controller::Controller(Model& model, View& view, bool* pKeepRunning)
: m_model(model), m_view(view), m_pKeepRunning(pKeepRunning), m_spaceDown(false), 
    m_rand(12),m_Q(nullptr),m_replayQue(),m_qFactors(), si(),
    sj(),qIn()
{
    // init some q-learning parameters
    iter = 0; // used to iterate through m_replayQue
    exSt = 0; 
    numActions = 11;
    epsilon = 0.5;
    alphaK = 1.0;
    gamma = 0.95;
    learningRate = 0.03;

    // initialize some member variables
    stateSize = 4;
    bufSize = 30;
    m_state.resize(stateSize);
    si.resize(stateSize);
    sj.resize(stateSize);
    qIn.resize(stateSize+1);
    getState(m_model,m_state);
    m_training = false;
    m_replayOut = false;
    m_greedy = false;
    tim = 0;
    life = 0;

    // setup the neuralnet
	if(access("model.json", 0) == 0) // If the file "model.json" exists...
	{
		// Load the existing model from the file
		cout << "Loading the model from file...\n";
		Json doc;
		doc.loadJson("model.json");
		m_Q = new NeuralNet(*doc.root());
	}
	else
	{
		// Make a new model
		cout << "Making a new model...\n";
		m_Q = new NeuralNet();
		m_Q->layers().push_back(new LayerLinear(5, 10));
		m_Q->layers().push_back(new LayerTanh(10));
		m_Q->layers().push_back(new LayerLinear(10, 8));
		m_Q->layers().push_back(new LayerTanh(8));
		m_Q->layers().push_back(new LayerLinear(8, 1));
		m_Q->layers().push_back(new LayerTanh(1));
		m_Q->init_weights(m_rand);
    }

    // setup replay queue
	if(access("inbuff.arff", 0) == 0) 
	{
        m_replayQue.loadARFF("inbuff.arff");
	}
	else
	{
        m_replayQue.setSize(bufSize,stateSize+1);
		// Make a new buffer
        for (size_t i = 0; i < bufSize; i++)
        {
            m_replayQue[i][0] = m_state[0];
            m_replayQue[i][1] = m_state[1];
            m_replayQue[i][2] = m_state[2];
            m_replayQue[i][3] = m_state[3];
            m_replayQue[i][4] = (double)m_rand.next(9)/4.0 - 1.0;
        }
    }
	if(access("qfbuff.arff", 0) == 0) 
	{
        m_qFactors.loadARFF("qfbuff.arff");
	}
	else
	{
		// Make a new buffer
        m_qFactors.setSize(bufSize,1);
        for (size_t i = 0; i < bufSize; i++)
            m_qFactors[i][0] = m_rand.uniform()-0.5;
    }
}

// This is the destructor
Controller::~Controller()
{
    cout << "time counter = " << tim << std::endl;
    // Save the model to a file
    cout << "Saving neuralnet to file...\n";
    Json doc;
    JsonNode* pNode = m_Q->marshal(doc);
    pNode->saveJson("model.json");
    // save the replay buffer to a file
    m_replayQue.saveARFF("inbuff.arff");
    m_qFactors.saveARFF("qfbuff.arff");

    delete(m_Q);
}

void Controller::update()
{
	SDL_Event event;
	while(SDL_PollEvent(&event))
	{
		if(event.type == SDL_QUIT)
			*m_pKeepRunning = false;
	}
	const Uint8* keys = SDL_GetKeyboardState(NULL);
	if(keys[SDL_SCANCODE_ESCAPE])
		*m_pKeepRunning = false;
	int mouseX, mouseY;
	Uint32 mouse_buttons = SDL_GetMouseState(&mouseX, &mouseY);
	if(keys[SDL_SCANCODE_SPACE])
	{
		if(!m_spaceDown)
		{
			m_spaceDown = true;
			m_view.visualizing = !m_view.visualizing; // Toggle the display on/off. (Training is much faster when off.)
			if(m_view.visualizing)
				cout << "Visualize mode (slow)\n";
			else
				cout << "No-visualize mode (fast)\n";
			cout.flush();
		}
	}
	else
		m_spaceDown = false;
	m_model.applyForce(0.0);
	if(keys[SDL_SCANCODE_RIGHT])
	{
		m_model.applyForce(0.1);
	}
	if(keys[SDL_SCANCODE_LEFT])
	{
		m_model.applyForce(-0.1);
	}
	if(keys[SDL_SCANCODE_T])
	{
        m_training = !m_training;
        if(m_training)
            cout << "Training!\n";
        else
            cout << "Not Training!\n";
	}
	if(keys[SDL_SCANCODE_R])
	{
        m_replayOut = !m_replayOut;
        if(m_replayOut)
            cout << "replayOut!\n";
        else
            cout << "don't replayOut!\n";
	}
	if(mouse_buttons & SDL_BUTTON(SDL_BUTTON_LEFT))
    {
		//m_model.onClick(mouseX, mouseY);
    }


    // --------------------------------------------
    // implement q-learning
    // --------------------------------------------

    // Pick an action
    getState(m_model,si);
	if(m_training && m_rand.uniform() < epsilon)
	{
		// Explore (pick a random action)
		action = m_rand.next(11);
	}
	else
	{
		// Exploit (pick the best action)
		action = 0;
        buildQin(si,action,qIn);
        const Vec& pred = m_Q->forwardprop(qIn);
        qFact_a = pred[0];
		for(int candidate = 1; candidate < numActions; candidate++)
        {
            buildQin(si,candidate,qIn);
            const Vec& p = m_Q->forwardprop(qIn);
            qFact_c = p[0];
			if(qFact_c > qFact_a)
            {
				action = candidate;
                qFact_a = qFact_c;
            }
        }
	}

	// Do the chosen action
    switch(action)
    {
        case 0: m_model.applyForce(1.0); break;
        case 1: m_model.applyForce(0.5); break;
        case 2: m_model.applyForce(0.1); break;
        case 3: m_model.applyForce(0.05); break;
        case 4: m_model.applyForce(0.01); break;
        case 5: m_model.applyForce(0.0); break;
        case 6: m_model.applyForce(-0.01); break;
        case 7: m_model.applyForce(-0.05); break;
        case 8: m_model.applyForce(-0.1); break;
        case 9: m_model.applyForce(-0.5); break;
        case 10: m_model.applyForce(-1.0); break;
    }
    // update model
    m_model.update();
    life++;

    // restart the game if the pole falls over
    if(std::abs(m_model.pole_angle) < M_PI/2.0)
    {
            m_model.reset();
            life = 0;
    }

    // get new state
    getState(m_model,sj);

    // ---------------------------
    // calculate q-factor update
    // ---------------------------

    // get the max q-factor for this state w/set of actions
    double a = action;
    action = 0;
    buildQin(sj,action,qIn);
    const Vec& pred = m_Q->forwardprop(qIn);
    qFact_a = pred[0];
    for(int candidate = 1; candidate < numActions; candidate++)
    {
        buildQin(sj,candidate,qIn);
        const Vec& p = m_Q->forwardprop(qIn);
        qFact_c = p[0];
        if(qFact_c > qFact_a)
        {
            action = candidate;
            qFact_a = qFact_c;
        }
    }

    // add q-factor to replay queue
    buildQin(si,a,qIn);
    double reward = std::abs(std::cos(m_model.pole_angle))/2.0;
    double qfct = (1.0-gamma)*alphaK*(reward+gamma*qFact_a);
    if(m_replayOut && tim%100==0)
    {
        cout << "#################################\n";
        for (size_t i=0;i<m_replayQue.rows();i++)
        {
            cout << "state/action=";
            m_replayQue[i].print(cout);
            cout << "\t\tqfactor=";
            m_qFactors[i].print(cout);
            cout << std::endl;
            cout.flush();
        }
    }
    addToQueue(iter,qIn,qfct);
    iter++;
    if(iter > bufSize-1) iter = 0;

    // update q-table i.e. train the neural net
    if(m_training)
    {
        int r = m_rand.next(bufSize);
        m_Q->train_incremental(m_replayQue[r],m_qFactors[r],learningRate);
        // swap trained pattern out with next pattern to be overwritten
        m_replayQue.swapRows(r,iter);
        m_qFactors.swapRows(r,iter);
    }

    // update timer count
    tim++;
    
    // adjust epsilon
    if(tim>2000)
    {
        epsilon = 0.6/std::sqrt((double)tim);
        exSt++;
    }
}

void Controller::getState(Model& mod,Vec& state)
{
    state[0] = mod.cart_position;
    state[1] = mod.cart_velocity/10.0;
    state[2] = mod.pole_angle/M_PI;
    state[3] = mod.pole_angular_velocity/(M_PI);
}

void Controller::buildQin(Vec& state, int act, Vec& out)
{
    out[0] = state[0];
    out[1] = state[1];
    out[2] = state[2];
    out[3] = state[3];
    out[4] = (double)act/4.0 - 1.0; // normalize to [-1,1]
}

void Controller::addToQueue(int it, Vec& qin, double qf)
{
    for (size_t i = 0; i<qin.size(); i++)
        m_replayQue[it][i] = qin[i];
    m_qFactors[it][0] = qf;
}
