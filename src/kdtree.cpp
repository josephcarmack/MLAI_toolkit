/*
 * kdtree.cpp
 * Copyright (C) 2017 joseph <joseph@JMC-WORKSTATION>
 *
 * Distributed under terms of the MIT license.
 */

#include "kdtree.h"
#include "vec.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using std::priority_queue;


KdTree::KdTree(const Matrix& points)
{
    // copy the data
    m_points.copyMetaData(points);
    m_points.copy(points);
    vector<size_t> indexes;
    // create a list of data point indexes (i.e.
    // row numbers from the data matrix).
    for(size_t i = 0; i < points.rows(); i++)
        indexes.push_back(i);

    // get data mins and maxes for data normalization.
    for(size_t i=0;i<m_points.cols();i++)
        if(!m_points.m_attrIsCateg[i])
        {
            double min = m_points.columnMin(i);
            double max = m_points.columnMax(i);
            mins.push_back(min);
            maxes.push_back(max);
        }
        else
        {
            mins.push_back(0.0);
            maxes.push_back(0.0);
        }

    // normalize the real valued data
    /* minMaxNormData(); */

    // compute the standard deviations of each feature
    for(size_t feat=0; feat<m_points.cols();feat++)
    {
        // ignore categorical features
        if(m_points.m_attrIsCateg.at(feat))
            standDev.push_back(0.0);
        else
        {
            double mean = m_points.columnMean(feat);
            double variance = 0.0;
            for(size_t i=0; i<m_points.rows(); i++)
            {
                double diff = m_points[i][feat] - mean;
                variance += diff*diff;
            }
            double STD = std::sqrt(variance/(m_points.rows()-1));
            standDev.push_back(STD);
        }
    }

    // build the kdTree
    m_pRoot = buildKdTree(indexes);

    // undo data normalization
    /* undoMinMaxNorm(); */
}



KdNode* KdTree::buildKdTree(vector<size_t>& indexes)
{
    size_t n = indexes.size();
    if(n<=8)
    {
        return new KdNodeLeaf(indexes);
    }
    else
    {
        // measure the mean in each dimension
        Vec means(m_points.cols());
        means.fill(0.0);
        for(size_t i=0;i<n;i++)
            for(size_t ftr=0;ftr<m_points.cols();ftr++)
                if(!m_points.m_attrIsCateg[ftr])
                    means[ftr] += m_points[indexes[i]][ftr];
        for(size_t ftr=0;ftr<m_points.cols();ftr++)
            if(!m_points.m_attrIsCateg[ftr])
                means[ftr] = means[ftr]/n;

        // calculate the deviation in each dimension
        Vec deviations(m_points.cols());
        deviations.fill(0.0);
        vector<vector<double>> perc;
        for(size_t ftr=0;ftr<m_points.cols();ftr++)
        {
            if(m_points.m_attrIsCateg[ftr])
            {
                vector<double> classes;
                // use normalized entropy for categorical values
                size_t k=m_points.valueCount(ftr);
                for(size_t cls=0;cls<k;cls++)
                {
                    int c=0;
                    for(size_t i=0;i<n;i++)
                        if(m_points[indexes[i]][ftr] == cls)
                            c++;
                    double h = (double)c/(double)n;
                    classes.push_back(h);
                    if(h==1.0) h = 0.0;
                    if(h>0)
                        h *= std::log(h)/std::log(k);
                    deviations[ftr] -= h;
                }
                perc.push_back(classes);
                // scale down the deviations so as to be more comparable to stand dev. 
                deviations[ftr] *= 0.45;
            }
            else
            {
                // use standard deviation for real values
                for(size_t i=0;i<n;i++)
                {
                    double dev = m_points[indexes[i]][ftr]-means[ftr];
                    deviations[ftr] += dev*dev;
                }
                deviations[ftr] = std::sqrt(deviations[ftr]/(n-1));
            }
        }

        // find dimension w/highest deviation & split data on it 
        size_t sDim = deviations.indexOfMax();

        // if dimension is categorical determine which one in perc array it is 
        size_t sDimCateg = m_points.m_attrIsCateg[sDim];
        size_t catInd = 0;
        size_t CLASS = 0;
        if(sDimCateg)
        {
            for(size_t i=0; i<sDim+1;i++)
                catInd += (int)m_points.m_attrIsCateg[i];
            catInd--;
        }
        std::vector<size_t> above;
        std::vector<size_t> below;
        if(sDimCateg)
        {
            // split on class with percentage closest to 50%
            size_t cls = 0;
            double prct = perc[(size_t)catInd][0];
            double dst = std::abs(prct -0.5);
            for(size_t c=1;c<perc[(size_t)catInd].size();c++)
            {
                double dst1 = std::abs(0.5 - perc[(size_t)catInd][c]);
                if(dst1 < dst)
                {
                    cls = c;
                    dst = dst1;
                }
            }
            CLASS = cls;
            // splitting the data
            for(size_t i=0;i<n;i++)
                if(m_points[indexes[i]][sDim] == cls)
                    below.push_back(indexes[i]);
                else
                    above.push_back(indexes[i]);
        }
        else
        {
            // slit on mean value for real valued features
            for(size_t i=0;i<n;i++)
                if(m_points[indexes[i]][sDim] < means[sDim])
                    below.push_back(indexes[i]);
                else
                    above.push_back(indexes[i]);
        }

        // spawn child nodes recursively
        KdNode* lesser = buildKdTree(below);
        KdNode* greater = buildKdTree(above);

        // join child nodes to interior node
        double value;
        if(sDimCateg)
            value = (double)CLASS;
        else
            value = means[sDim];//*(maxes[sDim]-mins[sDim])+mins[sDim];
        KdNode* outNode = new KdNodeInterior(lesser,greater,sDim,value,sDimCateg); 
        return outNode;
    }
}



void KdTree::findNeighbors(size_t k, const Vec& point,
        vector<size_t>& outNeighborIndexes,
        vector<double>& outWeights)
{
    // reset all stored node distances to zero
    resetNodeDist();
    // priority queue of points sorted by distance from point
    priority_queue<pair<double,size_t>,vector<pair<double,size_t>>,pComp> pointSet;    
    // priority queue of KdNodes sorted by distance from point
    priority_queue<pair<double,KdNode*>,vector<pair<double,KdNode*>>,nComp> nodeSet;    

    // add root node to node priority queue
    nodeSet.push(pair<double,KdNode*>(0.0,m_pRoot));
    while(!nodeSet.empty())
    {
        KdNode* n = nodeSet.top().second;
        double nDist = nodeSet.top().first;
        nodeSet.pop();
        if(n->isLeaf())
        {
            KdNodeLeaf* nn = dynamic_cast<KdNodeLeaf*>(n);
            // compute distances to all points in leaf node and
            // add to the points queue
            vector<size_t>& nP = nn->m_pointIndexes;
            
            for(size_t i=0;i<nP.size();i++)
            {
                double d = compDist(point,m_points[nP[i]]);
                pointSet.push(pair<double,size_t>(d,nP[i]));
            }
        }
        else
        {
            KdNodeInterior* nn = dynamic_cast<KdNodeInterior*>(n);
            // check if distance from point to n is greater than
            // the kth neighbor in the point queue
            if(!pointSet.empty() && pointSet.size()>=k)
            {
                double kthDist = 0.0;
                vector<pair<double,size_t>> putBack;
                for(size_t i=0;i<k;i++)
                {
                    kthDist = pointSet.top().first;
                    putBack.push_back(pointSet.top());
                    pointSet.pop();
                }
                for(size_t i=0;i<k;i++)
                    pointSet.push(putBack[i]);
                if(nDist > kthDist)
                    break;
            }

            // compute distances from child nodes to point
            KdNode* less = nn->less_than;
            KdNode* grtr = nn->greater_or_equal;
            double dl,dg; // for storing distances to child KdNodes
            compNodeDist(point,less,grtr,nn,dl,dg);

            // add child nodes to node priority queue
            nodeSet.push(pair<double,KdNode*>(dl,less));
            nodeSet.push(pair<double,KdNode*>(dg,grtr));
        }
    }

    // return the k-nearest point indexes & corresponding weights
    for(size_t i=0;i<k;i++)
    {
        outNeighborIndexes.push_back(pointSet.top().second);
        outWeights.push_back(1.0/pointSet.top().first);
        pointSet.pop();
    }
}




void KdTree::compNodeDist(const Vec& point,KdNode* lesser,KdNode* greater,KdNodeInterior* parent,double& outDLesser, double& outDGreater)
{
    // set child node distance to the distance of the parent node
    lesser->pointDist.copy(parent->pointDist);
    greater->pointDist.copy(parent->pointDist);
    // get split dim and value from parent node
    size_t sd = parent->sDim;
    double val = parent->value;
    if(parent->sDimCateg)
    {
        // check if point has same value as split feature
        if(point[sd] == (size_t)val)
            lesser->pointDist[sd] = 1.0;
        else
            greater->pointDist[sd] = 1.0;
    }
    else
    {
        // check if point is above split feature
        if(point[sd] >= val)
            lesser->pointDist[sd] = (point[sd] - val)/standDev[sd];
        else
            greater->pointDist[sd] = (point[sd] - val)/standDev[sd];
    }

    // calculate node distances
    outDLesser = lesser->pointDist.squaredMagnitude();
    outDGreater = greater->pointDist.squaredMagnitude();
}


void KdTree::minMaxNormData()
{
    for(size_t j=0;j<m_points.cols();j++)
    {
        if(!m_points.m_attrIsCateg[j])
            for(size_t i=0;i<m_points.rows();i++)
            {
                m_points[i][j] = (m_points[i][j]-mins[j]);
                m_points[i][j] = m_points[i][j]/(maxes[j]-mins[j]);
            }
    }

}



void KdTree::undoMinMaxNorm()
{
    for(size_t j=0;j<m_points.cols();j++)
    {
        if(!m_points.m_attrIsCateg[j])
            for(size_t i=0;i<m_points.rows();i++)
            {
                m_points[i][j] = m_points[i][j]*(maxes[j]-mins[j]);
                m_points[i][j] += mins[j];
            }
    }

}



double KdTree::compDist(const Vec& pointA, const Vec& pointB)
{
    if(pointA.size() != pointB.size())
        throw Ex("cannot compute distance between vectors with different sizes!");

    double dist = 0.0;
    for(size_t i=0; i<pointA.size(); i++)
    {
        if(m_points.m_attrIsCateg.at(i))
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


void KdTree::resetNodeDist()
{
    vector<KdNode*> nodes;
    nodes.push_back(m_pRoot);
    while(!nodes.empty())
    {
        KdNode* n = nodes.back();
        nodes.pop_back();
        n->pointDist.resize(m_points.cols());
        n->pointDist.fill(0.0);
        if(!n->isLeaf())
        {
            // add child nodes to nodes vec
            nodes.push_back(dynamic_cast<KdNodeInterior*>(n)->less_than);
            nodes.push_back(dynamic_cast<KdNodeInterior*>(n)->greater_or_equal);
        }
    }
}
