/*
 * kdtree.cpp
 * Copyright (C) 2017 joseph <joseph@JMC-WORKSTATION>
 *
 * Distributed under terms of the MIT license.
 */

#include "kdtree.h"
#include "vec.h"
#include <cmath> // for sqrt
#include <queue> // for priority_queue
#include <vector>

using std::vector;
using std::priority_queue;


KdTree::KdTree(const Matrix& points)
{
    // copy the data
    m_points.copyMetaData(points);
    m_points.copy(points);
    // create a list of data point indexes (i.e.
    // row numbers from the data matrix).
    for(size_t i = 0; i < points.rows(); i++)
        m_indexes.push_back(i);

    // compute the standard deviations of each feature
    for(size_t feat=0; feat<m_points.cols();feat++)
    {
        // ignore categorical features
        if(m_points.m_attrIsCateg.at(feat))
            m_standDev.push_back(0.0);
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
            m_standDev.push_back(STD);
        }
    }

    // build the kdTree
    m_pRoot = buildKdTree(m_indexes);
}



/**************************************************
 * Takes a vector of data point indexes (i.e. row
 * values from a Matrix object holding the data),
 * and splits the data into leaf nodes with
 * "leafLimit" number of data points per leaf. The
 * data is in each interior node is split on the
 * data dimension with the highest variance. Var-
 * iance for real valued data is measured via the
 * standard deviation. Variance for categorical 
 * valued data is measured using normalized
 * entropy.
**************************************************/

KdNode* KdTree::buildKdTree(vector<size_t>& indexes, size_t leafLimit)
{
    size_t n = indexes.size();
    if(n<=leafLimit)
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
            // split on mean value for real valued features
            for(size_t i=0;i<n;i++)
                if(m_points[indexes[i]][sDim] < means[sDim])
                    below.push_back(indexes[i]);
                else
                    above.push_back(indexes[i]);
        }

        // spawn child nodes recursively
        KdNode* lesser = buildKdTree(below,leafLimit);
        KdNode* greater = buildKdTree(above,leafLimit);

        // join child nodes to interior node
        double value;
        if(sDimCateg)
            value = (double)CLASS;
        else
            value = means[sDim];
        KdNode* outNode = new KdNodeInterior(lesser,greater,sDim,value,sDimCateg); 
        return outNode;
    }
}



/**************************************************
 * Traverses the kdTree to leaf nodes containing
 * data points closest to the data point of under
 * consideration. This is done using two priority
 * queues, one made of of nodes sorted by distance
 * to the data point and one of points sorted by
 * distance to the data point. The root node is
 * added to the node priority que to start. The
 * tree is then traversed by adding child nodes
 * to the node queue and leaf nodes point distri-
 * butions to the point queue. Searching stops
 * when the distance to the nearest child node
 * in the node queue is larger than the kth point
 * in the point queue.
 * ***********************************************/

void KdTree::findNeighbors(size_t k, const Vec& point,
        vector<size_t>& outNeighborIndexes,
        vector<double>& outDistances)
{
    // reset all stored node distances to zero
    resetNodeDist();

    // priority queue of points sorted by distance from point
    priority_queue<pair<double,size_t>,vector<pair<double,size_t>>,pComp> pointSet;    

    // priority queue of KdNodes sorted by distance from point
    priority_queue<pair<double,KdNode*>,vector<pair<double,KdNode*>>,nComp> nodeSet;    

    // add root node to node priority queue
    nodeSet.push(pair<double,KdNode*>(0.0,m_pRoot));

    // traverse the kdTree searching for k-nearest neighbors
    while(!nodeSet.empty())
    {
        // identify kdNode at the top of the node queue
        KdNode* n = nodeSet.top().second;
        double nDist = nodeSet.top().first;

        // remove the node from the queue
        nodeSet.pop();

        // if node is a leaf then add its points to the point queue
        if(n->isLeaf())
        {
            KdNodeLeaf* nn = dynamic_cast<KdNodeLeaf*>(n);
            // compute distances to all points in leaf node 
            vector<size_t>& nP = nn->m_pointIndexes;
            
            // add to the points queue
            for(size_t i=0;i<nP.size();i++)
            {
                double d = compDist(point,m_points[nP[i]]);
                pointSet.push(pair<double,size_t>(d,nP[i]));
            }
        }
        // if the node is an interior node, check if done searching.
        // If not done searching, add its child nodes to the node
        // queue
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
                    break; // break out of the while loop
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
        outDistances.push_back(pointSet.top().first);
        pointSet.pop();
    }
}



/**************************************************
 * Computes the distance between a data point and
 * the two child nodes of an interior kdNode. This
 * is needed to determine which child node to visit
 * next in order to find the next nearest neighbor
 * to the data point of interest.
 * ***********************************************/

void KdTree::compNodeDist(const Vec& point,KdNode* lesser,KdNode* greater,KdNodeInterior* parent,double& outDLesser, double& outDGreater)
{
    // set child node distance to the distance of the parent node
    lesser->pointDist.copy(parent->pointDist);
    greater->pointDist.copy(parent->pointDist);

    // get split dim and value from the parent node
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
            lesser->pointDist[sd] = (point[sd] - val)/m_standDev[sd];
        else
            greater->pointDist[sd] = (point[sd] - val)/m_standDev[sd];
    }

    // calculate and return node distances
    outDLesser = lesser->pointDist.squaredMagnitude();
    outDGreater = greater->pointDist.squaredMagnitude();
    outDLesser = sqrt(outDLesser);
    outDGreater = sqrt(outDGreater);
}



/**************************************************
 * Computes the distance between two data points.
 * For real valued data, Mahalanobis distance is
 * used. For categorical data, hamming distance is
 * used
 * ***********************************************/

double KdTree::compDist(const Vec& pointA, const Vec& pointB)
{
    if(pointA.size() != pointB.size())
        throw Ex("cannot compute distance between vectors with different sizes!");

    double dist = 0.0;
    for(size_t i=0; i<pointA.size(); i++)
    {
        if(m_points.m_attrIsCateg.at(i))
        {
            // use hamming distance for categorical data
            if(pointA[i] != pointB[i])
                dist += 1.0;
        }
        else
        {
            // use Mahalanobis distance for real valued data
            double diff = (pointA[i] - pointB[i])/m_standDev.at(i);
            dist += diff*diff;
        }
    }
    dist = std::sqrt(dist);
    return dist;
}



/**************************************************
 * Starts at the root node of the KdTree and 
 * travels down the tree zeroing out the poinDist
 * vector of each kdNode.
 * ***********************************************/

void KdTree::resetNodeDist()
{
    // add root node to vector of nodes that need zeroing
    vector<KdNode*> needsZeroing;
    needsZeroing.push_back(m_pRoot);
    while(!needsZeroing.empty())
    {
        KdNode* n = needsZeroing.back();
        needsZeroing.pop_back();
        n->pointDist.resize(m_points.cols());
        n->pointDist.fill(0.0);

        // if node is not a leaf, add its child nodes to the
        // vector of nodes that still needs zeroing
        if(!n->isLeaf())
        {
            KdNode* child;
            child = dynamic_cast<KdNodeInterior*>(n)->less_than;
            needsZeroing.push_back(child);
            child = dynamic_cast<KdNodeInterior*>(n)->greater_or_equal;
            needsZeroing.push_back(child);
        }
    }
}
