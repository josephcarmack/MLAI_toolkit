// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef NEURALNET_H
#define NEURALNET_H

#include "vec.h"
#include "matrix.h"
#include <vector>

class Rand;


/// An internal class used by the NeuralNet class
class Layer
{
public:
	Vec m_activation;
	Vec m_blame;

public:
	/// General-purpose constructor
	Layer(size_t out);

	/// Unmarshaling constructor
	Layer(const JsonNode& node);

	virtual ~Layer();

	size_t outputCount() { return m_activation.size(); }
	static Layer* unmarshal(const JsonNode& node);

	virtual JsonNode* marshal(Json& doc) = 0;
	virtual void init_weights(Rand& rand) = 0;
	virtual const Vec& forwardprop(const Vec& in) = 0;
	virtual void backprop(Vec& upStreamBlame) = 0;
	virtual void scale_gradient(double momentum) = 0;
	virtual void update_gradient(const Vec& in) = 0;
	virtual void step(double learning_rate) = 0;
	virtual void print() = 0;

protected:
	/// Helper method used by the marshal methods in the child classes
	JsonNode* makeJsonNode(Json& doc, const char* name);
};




/// A layer of linear weights
class LayerLinear : public Layer
{
	Matrix m_weights; // cols = in, rows = out
	Vec m_bias;
	Matrix m_weightDelta;
	Vec m_biasDelta;

public:
	/// General-purpose constructor
	LayerLinear(size_t in, size_t out);

	/// Unmarshaling constructor
	LayerLinear(const JsonNode& node);

	/// Destructor
	virtual ~LayerLinear();

	/// Returns the weights matrix of this layer
	Matrix& weights() { return m_weights; }

	/// Returns the bias vector of this layer
	Vec& bias() { return m_bias; }

	virtual JsonNode* marshal(Json& doc);
	virtual void init_weights(Rand& rand);
	virtual const Vec& forwardprop(const Vec& in);
	virtual void backprop(Vec& upStreamBlame);
	virtual void scale_gradient(double momentum);
	virtual void update_gradient(const Vec& in);
	virtual void step(double learning_rate);
	virtual void print();
};




/// A layer of tanh activation functions
class LayerTanh : public Layer
{
public:
	/// General-purpose constructor
	LayerTanh(size_t units);

	/// Unmarshaling constructor
	LayerTanh(const JsonNode& node);

	/// Destructor
	virtual ~LayerTanh();

	virtual JsonNode* marshal(Json& doc);
	virtual void init_weights(Rand& rand);
	virtual const Vec& forwardprop(const Vec& in);
	virtual void backprop(Vec& upStreamBlame);
	virtual void scale_gradient(double momentum);
	virtual void update_gradient(const Vec& in);
	virtual void step(double learning_rate);
	virtual void print();
};





/// A multi-layer neural network
class NeuralNet
{
private:
	std::vector<Layer*> m_layers;

public:
	/// General-purpose constructor
	NeuralNet();

	/// Unmarshaling constructor
	NeuralNet(const JsonNode& node);

	/// Destructor
	virtual ~NeuralNet();

	/// Marshals this object into a JSON DOM.
	JsonNode* marshal(Json& doc);

	/// Returns a reference to the  layers
	std::vector<Layer*>& layers() { return  m_layers; }

	/// Returns the number of units in the output layer of this neural network.
	size_t outputCount();

	/// Sets all the weights to small random values
	void init_weights(Rand& rand);

	/// Presents one pattern for incrementally refining the weights by stochastic gradient descent.
	void train_incremental(const Vec& in, const Vec& target, double learning_rate);

	/// Feeds a vector forward through the network.
	const Vec& forwardprop(const Vec& in);

	/// Computes the blame terms for all of the units in the output layer.
	void compute_output_layer_blame_terms(const Vec& target);

	/// Backpropagates the blame to compute the blame terms for all hidden layers.
	void backprop(Vec* pInputBlame = nullptr);

	/// Scales the gradient by the specified scalar.
	void scale_gradient(double scalar);

	/// Adds values from the most recent pattern presentation to the gradient.
	void update_gradient(const Vec& in);

	/// Takes a single step in the direction that should reduce error.
	void step(double learning_rate);

	/// Run a unit tests
	static void unit_test1();

	/// Print a representation of this neural net to stdout for debugging purposes
	void print();
};


#endif // NEURALNET_H
