
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/time.h> // for gettimeofday
#include "../../src/knn.h"
#include "../../src/kdtree.h"
#include "../../src/matrix.h"

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

double seconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char *argv[])
{

    //medical data problem
    Matrix trainFeat;
    trainFeat.loadARFF("data/hypo_train_features.arff");
    Matrix trainLab;
    trainLab.loadARFF("data/hypo_train_labels.arff");
    Matrix testFeat;
    testFeat.loadARFF("data/hypo_test_features.arff");
    Matrix testLab;
    testLab.loadARFF("data/hypo_test_labels.arff");

    // create k-nn classifier
    KNN classifier1(trainFeat,trainLab,true);

    // perform classification with knn and k=3 with linear weighting
    cout << "\n\nBeginning testing of the Hypothyroid dataset...\n";
    Matrix predictions;
    predictions.copyMetaData(testLab);
    predictions.copy(testLab);
    predictions.fill(0);
    Vec pred1(testLab.cols());
    double start = seconds();
    size_t numMisClass = 0;
    for(size_t i=0;i<testFeat.rows();i++)
    {
        classifier1.predict(3,testFeat[i],pred1);

        predictions[i].copy(pred1);
        // check if prediction is correct
        for(size_t j=0;j<testLab.cols();j++)
        {
            if(testLab[i][j] != predictions[i][j])
            {
                numMisClass++;
                break;
            }
        }
    }
    cout << "\nTesting complete...\n";

    // print run time and miss-classifications
    double dur = seconds() - start;
    cout << "\n------------------------------------------\n";
    cout << "run time = " << dur << " seconds.\n";
    cout << "------------------------------------------\n";
    cout << "\nThis is about 2X faster than the brute force\n";
    cout << "method for this data set. I would assume the\n";
    cout << "speedup to improve for larger data sets.\n\n";
    cout << "------------------------------------------\n";
    cout << "number of miss-classificiations  = " << numMisClass << endl;
    cout << "------------------------------------------\n\n";

	return 0;
}

