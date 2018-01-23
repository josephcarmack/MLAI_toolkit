// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "neuralnet.h"
#include "error.h"
#include "string.h"
#include "json.h"
#include "rand.h"
#include <math.h>
#include <cmath>
#include <iostream>

using std::vector;



Layer::Layer(size_t out)
{
	m_activation.resize(out);
	m_blame.resize(out);
}

Layer::Layer(const JsonNode& node)
{
	size_t units = node.field("units")->asInt();
	m_activation.resize(units);
	m_blame.resize(units);
}

Layer::~Layer()
{
}

Layer* Layer::unmarshal(const JsonNode& node)
{
	const char* name = node.field("type")->asString();
	if(strcmp(name, "linear") == 0)
		return new LayerLinear(node);
	else if(strcmp(name, "tanh") == 0)
		return new LayerTanh(node);
	else
		throw Ex("Unrecognized layer type: ", name);
}

JsonNode* Layer::makeJsonNode(Json& doc, const char* name)
{
	JsonNode* pNode = doc.newObj();
	pNode->addField(&doc, "type", doc.newString(name));
	pNode->addField(&doc, "units", doc.newInt(m_activation.size()));
	return pNode;
}








LayerLinear::LayerLinear(size_t in, size_t out)
: Layer(out)
{
	m_weights.setSize(out, in);
	m_weightDelta.setSize(out, in);
	m_bias.resize(out);
	m_biasDelta.resize(out);
}

LayerLinear::LayerLinear(const JsonNode& node)
: Layer(node), m_weights(*node.field("weights")), m_bias(*node.field("bias"))
{
	if(m_weights.rows() != m_bias.size())
		throw Ex("size mismatch");
	m_weightDelta.setSize(m_weights.rows(), m_weights.cols());
    m_biasDelta.resize(m_bias.size());
}

LayerLinear::~LayerLinear()
{
}

JsonNode* LayerLinear::marshal(Json& doc)
{
	JsonNode* pNode = makeJsonNode(doc, "linear");
	pNode->addField(&doc, "weights", m_weights.marshal(doc));
	pNode->addField(&doc, "bias", m_bias.marshal(doc));
	return pNode;
}

void LayerLinear::init_weights(Rand& rand)
{
	double mag = std::max(0.03, 1.0 / m_weights.cols());
	for(size_t j = 0; j < m_weights.rows(); j++)
	{
		for(size_t i = 0; i < m_weights.cols(); i++)
			m_weights[j][i] = mag * rand.normal();
		m_bias[j] = mag * rand.normal();
	}
	m_weightDelta.fill(0.0);
	m_biasDelta.fill(0.0);
}

const Vec& LayerLinear::forwardprop(const Vec& in)
{
	if(in.size() != m_weights.cols())
		throw Ex("size mismatch");
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_activation[i] = in.dotProduct(m_weights[i]) + m_bias[i];
	return m_activation;
}

void LayerLinear::backprop(Vec& upStreamBlame)
{
	for(size_t i = 0; i < upStreamBlame.size(); i++)
	{
		double e = 0.0;
		for(size_t j = 0; j < m_weights.rows(); j++)
			e += m_weights[j][i] * m_blame[j];
		upStreamBlame[i] = e;
	}
}

void LayerLinear::scale_gradient(double momentum)
{
	m_weightDelta *= momentum;
	m_biasDelta *= momentum;
}

void LayerLinear::update_gradient(const Vec& in)
{
	for(size_t j = 0; j < m_weights.rows(); j++)
	{
		Vec& wd = m_weightDelta[j];
		for(size_t i = 0; i < m_weights.cols(); i++)
			wd[i] += in[i] * m_blame[j];
		m_biasDelta[j] += m_blame[j];
	}
}

void LayerLinear::step(double learning_rate)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		Vec& w = m_weights[i];
		Vec& wd = m_weightDelta[i];
		for(size_t j = 0; j < m_weights.cols(); j++)
			w[j] += learning_rate * wd[j];
		m_bias[i] += learning_rate * m_biasDelta[i];
	}
}

void LayerLinear::print()
{
	std::cout << "Bias: " << to_str(m_bias) << "\n";
	std::cout << "Weights: " << to_str(m_weights) << "\n";
}










LayerTanh::LayerTanh(size_t units)
: Layer(units)
{
}

LayerTanh::LayerTanh(const JsonNode& node)
: Layer(node)
{
}

LayerTanh::~LayerTanh()
{
}

JsonNode* LayerTanh::marshal(Json& doc)
{
	JsonNode* pNode = makeJsonNode(doc, "tanh");
	return pNode;
}

void LayerTanh::init_weights(Rand& rand)
{
}

const Vec& LayerTanh::forwardprop(const Vec& in)
{
	for(size_t i = 0; i < m_activation.size(); i++)
	{
		if(in[i] >= 700.0)
			m_activation[i] = 1.0;
		else if(in[i] < -700.0)
			m_activation[i] = -1.0;
		else m_activation[i] = tanh(in[i]);
	}
	return m_activation;
}

void LayerTanh::backprop(Vec& upStreamBlame)
{
	for(size_t i = 0; i < upStreamBlame.size(); i++)
	{
		upStreamBlame[i] = m_blame[i] * (1.0 - m_activation[i] * m_activation[i]);
	}
}

void LayerTanh::scale_gradient(double momentum)
{
}

void LayerTanh::update_gradient(const Vec& in)
{
}

void LayerTanh::step(double learning_rate)
{
}

void LayerTanh::print()
{
}








NeuralNet::NeuralNet()
{
}

NeuralNet::NeuralNet(const JsonNode& node)
{
	JsonListIterator it(&node);
	while(it.remaining() > 0)
	{
		m_layers.push_back(Layer::unmarshal(*it.current()));
		it.advance();
	}
}

// virtual
NeuralNet::~NeuralNet()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
}

JsonNode* NeuralNet::marshal(Json& doc)
{
	JsonNode* pNode = doc.newList();
	for(size_t i = 0; i < m_layers.size(); i++)
		pNode->addItem(&doc, m_layers[i]->marshal(doc));
	return pNode;
}

size_t NeuralNet::outputCount()
{
	return m_layers[m_layers.size() - 1]->outputCount();
}

void NeuralNet::init_weights(Rand& rand)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->init_weights(rand);
}

const Vec& NeuralNet::forwardprop(const Vec& in)
{
	m_layers[0]->forwardprop(in);
	for(size_t i = 1; i < m_layers.size(); i++)
		m_layers[i]->forwardprop(m_layers[i - 1]->m_activation);
	return m_layers[m_layers.size() - 1]->m_activation;
}

void NeuralNet::compute_output_layer_blame_terms(const Vec& target)
{
	Layer& output_layer = *m_layers[m_layers.size() - 1];
	for(size_t i = 0; i < target.size(); i++)
		output_layer.m_blame[i] = target[i] - output_layer.m_activation[i];
}

void NeuralNet::backprop(Vec* pInputBlame)
{
	for(size_t i = m_layers.size() - 1; i > 0; i--)
		m_layers[i]->backprop(m_layers[i - 1]->m_blame);
	if(pInputBlame)
		m_layers[0]->backprop(*pInputBlame);
}

void NeuralNet::scale_gradient(double scalar)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->scale_gradient(scalar);
}

void NeuralNet::update_gradient(const Vec& in)
{
	const Vec* pIn = &in;
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		m_layers[i]->update_gradient(*pIn);
		pIn = &m_layers[i]->m_activation;
	}
}

void NeuralNet::step(double learning_rate)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->step(learning_rate);
}

void NeuralNet::train_incremental(const Vec& in, const Vec& target, double learning_rate)
{
	scale_gradient(0.0);
	forwardprop(in);
	compute_output_layer_blame_terms(target);
	backprop();
	update_gradient(in);
	step(learning_rate);
}

void NeuralNet::print()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		std::cout << "Layer " << to_str(i) << ":\n";
		m_layers[i]->print();
	}
}

void NeuralNet::unit_test1()
{
	Rand rand(0);
	NeuralNet mlp;
	LayerLinear* l1 = new LayerLinear(2, 3);
	LayerTanh* a1 = new LayerTanh(3);
	LayerLinear* l2 = new LayerLinear(3, 2);
	LayerTanh* a2 = new LayerTanh(2);
	mlp.layers().push_back(l1);
	mlp.layers().push_back(a1);
	mlp.layers().push_back(l2);
	mlp.layers().push_back(a2);
	Matrix features(1, 2);
	Matrix labels(1, 2);
	features[0][0] = 0.3;
	features[0][1] = -0.2;
	labels[0][0] = 0.1;
	labels[0][1] = 0.0;
	mlp.init_weights(rand);

	// Set the weights
	Matrix& w1 = l1->weights();
	w1[0][0] = 0.1; w1[0][1] = 0.1;
	w1[1][0] = 0.0; w1[1][1] = 0.0;
	w1[2][0] = 0.1; w1[2][1] = -0.1;
	Vec& b1 = l1->bias();
	b1[0] = 0.1;
	b1[1] = 0.1;
	b1[2] = 0.0;
	Matrix& w2 = l2->weights();
	w2[0][0] = 0.1; w2[0][1] = 0.1; w2[0][2] = 0.1;
	w2[1][0] = 0.1; w2[1][1] = 0.3; w2[1][2] = -0.1;
	Vec& b2 = l2->bias();
	b2[0] = 0.1;
	b2[1] = -0.2;

	// Train
	mlp.train_incremental(features[0], labels[0], 0.1);

	// Spot check the activations
	if(std::abs(0.09966799462495 - a1->m_activation[1]) > 1e-8)
		throw Ex("act1 wrong");
	if(std::abs(-0.16268123406035 - a2->m_activation[1]) > 1e-8)
		throw Ex("act2 wrong");

	// Spot check the blames
	if(std::abs(0.15837584528136 - l2->m_blame[1]) > 1e-8)
		throw Ex("blame1 wrong");
	if(std::abs(0.04457938080482 - l1->m_blame[1]) > 1e-8)
		throw Ex("blame2 wrong");

	// Spot check the updated weights
	if(std::abs(-0.0008915876160964 - l1->weights()[1][1]) > 1e-8)
		throw Ex("weight1 wrong");
	if(std::abs(0.30157850028962 - l2->weights()[1][1]) > 1e-8)
		throw Ex("weight2 wrong");
}

