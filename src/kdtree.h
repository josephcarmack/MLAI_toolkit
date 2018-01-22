/*
 * kdtree.h
 * Copyright (C) 2017 joseph <joseph@JMC-WORKSTATION>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include "matrix.h"
#include "vec.h"

using std::vector;
using std::pair;

/**************************************************
 * base kdnode class. Child classes are interior
 * nodes and leaf nodes
 * ***********************************************/

class KdNode
{
    public:
        // Vector for storing distances of a point
        // in the data set to this node
        Vec pointDist;

        // methods
        virtual ~KdNode() {}
        virtual bool isLeaf() = 0;
};


/**************************************************
 * Interior kdnode class
 * ***********************************************/

class KdNodeInterior : public KdNode
{
    public:
        // Pointers to child nodes
        KdNode* less_than;
        KdNode* greater_or_equal;

        // flag indicating if tree split dimension is
        // categorical or not.
        bool sDimCateg;

        // actual split dimension
        size_t sDim;

        // value of the split dimension
        double value;

        // methods
        virtual ~KdNodeInterior()
        {
            delete(less_than);
            delete(greater_or_equal);
        }

        // constructor: takes pointers to child nodes and values
        // for members as input
        KdNodeInterior(KdNode* left, KdNode* right,size_t d,double v,bool categ)
            : less_than(left), greater_or_equal(right)
        {
            sDim = d;
            value = v;
            sDimCateg = categ;
        }

        // function for querying leaf node status
        virtual bool isLeaf() { return false; };
};



/**************************************************
 * leaf kdnode class
 * ***********************************************/

class KdNodeLeaf : public KdNode
{
    public:
        // Vector of data points (just their indexes)
        // associated with this leaf node
        vector<size_t> m_pointIndexes;

        // destructor and constructor
        virtual ~KdNodeLeaf() {}
        KdNodeLeaf(vector<size_t> indexes)
            : m_pointIndexes(indexes)
        {};

        // function for querying leaf node status
        virtual bool isLeaf() { return true; };
};




/**************************************************
 * comparison struct for sorting data points by
 * distance to another point.
 * ***********************************************/

struct pComp {
    bool operator()(pair<double,size_t>& lhs,pair<double,size_t>& rhs)const
    {
        return lhs.first > rhs.first;
    }
};




/**************************************************
 * comparison struct for sorting kdNodes by 
 * distance to a data point.
 * ***********************************************/

struct nComp
{
    bool operator()(pair<double,KdNode*>& lhs,pair<double,KdNode*>& rhs)const
    {
        return lhs.first > rhs.first;
    }
};




/**************************************************
 * Actual KdTree class which builds the entire
 * tree composed of interior and leaf kdNode
 * objects. It takes a dataset as input and can 
 * find k-nearest neighbors of that dataset in an
 * efficient manner.
 * ***********************************************/

class KdTree
{
    private:
        // a copy of the model data
        Matrix m_points;
        
        // vector of data standard deviations
        std::vector<double> m_standDev;

        // methods
        static bool indexCompare(size_t ind1,size_t ind2,size_t sd);

        // resets all the "pointDist" Vector components to zero
        void resetNodeDist();

        // computes distances between an input data point and two
        // kdNodes and stores the results in outDLesser & outDGreater
        void compNodeDist(const Vec& point,KdNode* lesser,KdNode* greater,KdNodeInterior* parent,double& outDLesser,double& outDGreater);

        // computes the distance between two data points
        double compDist(const Vec& pointA, const Vec& pointB);

    public:
        // pointer to the root node in the kdTree
        KdNode* m_pRoot;

        // vector of row indexes for the data
        vector<size_t> m_indexes;

        // parameter that determines the number of data points
        // stored per leaf node of the KdTree
        size_t leafLimit;

        // constructor and destructor
        KdTree(const Matrix& points);
        ~KdTree()
        {
            delete(m_pRoot);
        }

        // takes a vector of data point indexes and builds the
        // KdTree with leaf nodes containing "leafLimit" number
        // of data points.
        KdNode* buildKdTree(vector<size_t>& indexes, size_t leafLimit=8);

        // finds the "k" nearest neighbors to the input data point
        // and stores the neighbors indexes and distances in 
        // outNeighborIndexes and outDistances
        void findNeighbors(size_t k, const Vec& point,
                vector<size_t>& outNeighborIndexes,
                vector<double>& outDistances);
};

#endif /* !KDTREE_H */
