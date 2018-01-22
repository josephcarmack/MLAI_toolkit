/*
 * knn.cpp
 * Copyright (C) 2017 joseph Carmack
 *
 * Distributed under terms of the MIT license.
 */

#include "knn.h"


/*****************************************************
 * Constructor
 *****************************************************/

KNN::KNN(const Matrix& inFeat, const Matrix& inLab, size_t inLeafLimit)
    : \
            m_modelFeat(inFeat),\
            m_modelLab(inLab),\
            myTree(inFeat)
{
    leafLimit = inLeafLimit;
}



/*****************************************************
 * Destructor
 *****************************************************/

KNN::~KNN()
{
}



/*****************************************************
 * Make a prediction using k-nearest neighbors. For 
 * real valued data, the computed label is an average
 * of its k-nearest neighbor values linearly weighted
 * by the distance of each neighbor to the point. For
 * Categorical data, the label is computed as the most
 * frequent label found amoung its k-nearest neighbors
 * where frequency is weighted by distance to the data 
 * point.
 *****************************************************/

void KNN::predict(size_t k, const Vec& inFeat, Vec& outLab)
{
    // resize the outLab Vec to appropriate size
    outLab.resize(m_modelLab.cols());

    // find k-nearest neighbors
    std::vector<size_t> indices;
    std::vector<double> distances;
    myTree.findNeighbors(k,inFeat,indices,distances);

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
                catFreq[val] += 1.0/distances[nbr];
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
            // compute weighted mean (linear weighting)
            double mean = 0.0;
            double weightSum = 0.0;
            for(size_t nbr=0; nbr<k; nbr++)
            {
                mean += m_modelLab[indices[nbr]][lab];
                weightSum += 1.0/distances[nbr];
            }
            mean = mean/weightSum;
            outLab[lab] = mean;
        }
    }
}



/*****************************************************
 * Resets the leafLimit for the kdTree storing the 
 * model data and rebuilds the kdTree.
 *****************************************************/

void KNN::setLeafLimit(size_t inLeafLimit)
{
    myTree.leafLimit = inLeafLimit;
    myTree.m_pRoot = myTree.buildKdTree(myTree.m_indexes);
}
