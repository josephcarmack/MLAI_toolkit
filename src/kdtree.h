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
#include <utility>

using std::vector;
using std::pair;

/**************************************************
 * base kdnode class. Child classes are interior
 * nodes and leaf nodes
 * ***********************************************/

class KdNode
{
    public:
        // members
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
        // members
        KdNode* less_than;
        KdNode* greater_or_equal;
        bool sDimCateg;
        size_t sDim;
        double value;

        // methods
        virtual ~KdNodeInterior()
        {
            delete(less_than);
            delete(greater_or_equal);
        }

        KdNodeInterior(KdNode* left, KdNode* right,size_t d,double v,bool categ)
            : less_than(left), greater_or_equal(right)
        {
            sDim = d;
            value = v;
            sDimCateg = categ;
        }

        virtual bool isLeaf() { return false; };
};



/**************************************************
 * leaf kdnode class
 * ***********************************************/

class KdNodeLeaf : public KdNode
{
    public:
        // members
        vector<size_t> m_pointIndexes;

        // methods
        virtual ~KdNodeLeaf() {}
        KdNodeLeaf(vector<size_t> indexes)
            : m_pointIndexes(indexes)
        {};
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
 * comparison struct for sorting kdnodes by 
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
 * tree composed of interior and leaf kdnode
 * objects. It takes a dataset as input and can 
 * find k-nearest neighbors of that dataset in an
 * efficient manner.
 * ***********************************************/

class KdTree
{
    private:
        Matrix m_points;
        KdNode* m_pRoot;
        std::vector<double> mins;
        std::vector<double> maxes;
        std::vector<double> standDev;

        // methods
        static bool indexCompare(size_t ind1,size_t ind2,size_t sd);
        void resetNodeDist();
        void compNodeDist(const Vec& point,KdNode* lesser,KdNode* greater,KdNodeInterior* parent,double& outDLesser,double& outDGreater);

    public:
        KdTree(const Matrix& points);
        ~KdTree()
        {
            delete(m_pRoot);
        }

        KdNode* buildKdTree(vector<size_t>& indexes);

        void findNeighbors(size_t k, const Vec& point,
                vector<size_t>& outNeighborIndexes,
                vector<double>& outWeights);
        void minMaxNormData();
        void undoMinMaxNorm();
        double compDist(const Vec& pointA, const Vec& pointB);
};

#endif /* !KDTREE_H */
