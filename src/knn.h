/*
 * knn.h
 * Copyright (C) 2017 joseph <joseph@JMC-WORKSTATION>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef KNN_H
#define KNN_H

# include <utility>
# include <algorithm> // for sort
# include "matrix.h"
# include "vec.h"
# include "error.h"
# include "kdtree.h"

class KNN
{
    public:
        // members
        const Matrix& m_modelFeat;
        const Matrix& m_modelLab;

        //methods
        KNN(const Matrix& inFeat, const Matrix& inLab,bool kdtree=true);
        ~KNN();
        void predict(size_t k,const Vec& inFeat, Vec& outLab);

    private:
        // members
        std::vector<double> standDev;
        KdTree myTree;
        bool useKdTree;

        // methods
        double computeDistance(const Vec& pointA, const Vec& pointB);

        // find neighbors and weights
        void findNeighborsByBruteForce(
                const Vec& point,
                const Matrix& data,
                size_t k, 
                std::vector<size_t>& outIndexes,
                std::vector<double>& weights
                );

        // method for sorting list of neighbors by distance
        static bool myComparator(const std::pair<size_t,double>& a,
                const std::pair<size_t,double>& b);
};

#endif /* !KNN_H */
