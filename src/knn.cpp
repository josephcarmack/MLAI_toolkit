/*
 * knn.cpp
 * Copyright (C) 2017 joseph <joseph@JMC-WORKSTATION>
 *
 * Distributed under terms of the MIT license.
 */

#include "knn.h"
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;

/*****************************************************
 * Constructor
 *****************************************************/

KNN::KNN(const Matrix& inFeat, const Matrix& inLab, bool kdtree)
    : m_modelFeat(inFeat), m_modelLab(inLab),myTree(inFeat)
{
    useKdTree = kdtree;
    // compute the standard deviations of each feature
    for(size_t feat=0; feat<inFeat.cols();feat++)
    {
        // ignore categorical features
        if(inFeat.m_attrIsCateg.at(feat))
        {
            standDev.push_back(0.0);
        }
        else
        {
            double mean = inFeat.columnMean(feat);
            double variance = 0.0;
            for(size_t i=0; i<inFeat.rows(); i++)
            {
                double diff = inFeat[i][feat] - mean;
                variance += diff*diff;
            }
            double STD = std::sqrt(variance/(inFeat.rows()-1));
            standDev.push_back(STD);
        }
    }
}



/*****************************************************
 * Destructor
 *****************************************************/

KNN::~KNN()
{
}



/*****************************************************
 * make a prediction using k-nearest neighbors
 *****************************************************/

void KNN::predict(size_t k, const Vec& inFeat, Vec& outLab)
{
    // resize the outLab Vec
    outLab.resize(m_modelLab.cols());

    // find k-nearest neighbors
    std::vector<size_t> indices;
    std::vector<double> weights;
    if(useKdTree)
        myTree.findNeighbors(k,inFeat,indices,weights);
    else
        findNeighborsByBruteForce(inFeat,m_modelFeat,k,indices,weights);

    // compute the return label Vec by averaging for real values
    // and using highest frequency for categorical values 
    for(size_t lab=0; lab<outLab.size(); lab++)
    {
        if(m_modelLab.m_attrIsCateg.at(lab))
        {
            // find most frequent category of the neighbors
            std::vector<double> catFreq;
            size_t numCat = m_modelLab.valueCount(lab);
            for(size_t i=0; i<numCat;i++)
                catFreq.push_back(0);
            for(size_t nbr=0; nbr<k; nbr++)
            {
                size_t val = m_modelLab[indices[nbr]][lab];
                catFreq[val] += weights[nbr];
            }
            std::pair<size_t,double> ind_max(0,catFreq[0]);
            for(size_t i=1; i<numCat;i++)
                if(catFreq[i] > ind_max.second)
                {
                    ind_max.second = catFreq[i];
                    ind_max.first = i;
                }
            // assign chosen value
            outLab[lab] = ind_max.first;
        }
        else
        {
            // compute weighted mean
            double mean = 0.0;
            double weightSum = 0.0;
            for(size_t nbr=0; nbr<k; nbr++)
            {
                mean += m_modelLab[indices[nbr]][lab];
                weightSum += weights[nbr];
            }
            mean = mean/weightSum;
            outLab[lab] = mean;
        }
    }
}



/*****************************************************
 * Brute force method for finding the k-nearest
 * neighbors of the input point which is a feature
 * vector.
 *****************************************************/

void KNN::findNeighborsByBruteForce(
        const Vec& point,
        const Matrix& data,
        size_t k,
        std::vector<size_t>& outIndexes,
        std::vector<double>& weights
        )
{
    std::vector< std::pair < size_t,double > > index_dist;
    for(size_t i=0; i< data.rows(); i++)
    {
        double dist = computeDistance(point, data[i]);
        index_dist.push_back(std::pair<size_t,double>(i,dist));
    }
    std::sort(index_dist.begin(),index_dist.end(),myComparator);
    outIndexes.resize(k);
    weights.resize(k);
    // return k-nearest neighbors and weights
    for(size_t i=0; i<k; i++)
    {
        outIndexes[i] = index_dist[i].first;
        weights[i] = 1.0/index_dist[i].second;
    }
}



/*****************************************************
 * compute the distance between two feature vectors.
 * Uses Mahalanobis distance for real values and 
 * Hamming distance for categorical values
 *****************************************************/

double KNN::computeDistance(const Vec& pointA, const Vec& pointB)
{
    if(pointA.size() != pointB.size())
        throw Ex("cannot compute distance between vectors with different sizes!");

    double dist = 0.0;
    for(size_t i=0; i<pointA.size(); i++)
    {
        if(m_modelFeat.m_attrIsCateg.at(i))
        {
            // use hamming distance
            if(pointA[i] != pointB[i])
                dist += 1.0;
        }
        else
        {
            // use Mahalanobis distance
            double diff = (pointA[i] - pointB[i])/standDev.at(i);
            dist += diff*diff;
        }
    }
    dist = std::sqrt(dist);
    return dist;
}



/*****************************************************
 * A comparison method for sorting neighbor distances.
 *****************************************************/

bool KNN::myComparator(const std::pair<size_t,double>& a,
        const std::pair<size_t,double>& b)
{
    return (a.second < b.second);
}
