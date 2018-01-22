/*
 * knn.h
 * Copyright (C) 2017 joseph Carmack
 *
 * Distributed under terms of the MIT license.
 */

#ifndef KNN_H
#define KNN_H

# include "matrix.h"
# include "vec.h"
# include "kdtree.h"

class KNN
{
    public:
        // data that makes up the knn model
        const Matrix& m_modelFeat;
        const Matrix& m_modelLab;

        // constructor and destructor
        KNN(const Matrix& inFeat, const Matrix& inLab, size_t inLeafLimit=8);
        ~KNN();

        // make a prediction using "k" neighbors. "inFeat"
        // is the input feature vector and "outLab" is where
        // the prediction is stored.
        void predict(size_t k,const Vec& inFeat, Vec& outLab);

        // changes the number of points per leaf of the kdTree
        // and rebuilds the kdTree accordingly
        void setLeafLimit(size_t inLeafLimit);

    private:
        // Put data in kdTree for efficiently finding neighbors
        KdTree myTree; 

        // parameter that determines the number of data points
        // stored per leaf node of the KdTree
        size_t leafLimit;
};

#endif /* !KNN_H */
