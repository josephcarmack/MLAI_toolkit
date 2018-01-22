
#include <exception>
#include <iostream>
#include <sys/time.h> // for gettimeofday
#include "../../src/knn.h"
#include "../../src/kdtree.h"
#include "../../src/matrix.h"

using std::cerr;
using std::cout;
using std::endl;


// function for timing computations
double seconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char *argv[])
{
    //-------------------------------------------------
    // Load the hypothyroid data stored in the "data"
    // directory into Matrix objects.
    // ------------------------------------------------

    Matrix trainFeat;
    trainFeat.loadARFF("data/hypo_train_features.arff");
    Matrix trainLab;
    trainLab.loadARFF("data/hypo_train_labels.arff");
    Matrix testFeat;
    testFeat.loadARFF("data/hypo_test_features.arff");
    Matrix testLab;
    testLab.loadARFF("data/hypo_test_labels.arff");

    // ------------------------------------------------
    // Create a k-nearest-neighbor classifier: the knn
    // object takes the training data as its model.
    // ------------------------------------------------

    size_t k = 3; // number of nearest neighbors to use in KNN model
    size_t pointsPerLeaf = 8; // the number of points stored per leaf for the KNN's KdTree
    KNN classifier(trainFeat,trainLab,pointsPerLeaf);

    // ------------------------------------------------
    // perform classification with knn and k=3 with 
    // linear weighting
    // ------------------------------------------------
    
    cout << "\n\nBeginning testing of the Hypothyroid dataset using k-nearest neighbor classifier...\n";

    // Vector for storing predictions
    Vec pred(testLab.cols());

    // for storing the number of miss-classifications
    size_t numMisClass = 0;

    // start testing
    double start = seconds();
    for(size_t i=0;i<testFeat.rows();i++)
    {
        classifier.predict(k,testFeat[i],pred);

        // check if prediction is correct
        for(size_t j=0;j<testLab.cols();j++)
        {
            if(testLab[i][j] != pred[j])
            {
                numMisClass++;
                break;
            }
        }
    }
    double dur = seconds() - start;

    // print test results
    cout << "KNN model testing complete:\n";
    cout << "\n--------------------------------------------\n";
    cout << "TESTING RESULTS:\n";
    cout << "--------------------------------------------\n";
    cout << "KNN using k = " << k << " and linear weighting\n";
    cout << "testing time = " << dur << " seconds.\n";
    cout << "percentage of miss-classificiations  = ";
    cout << (double)numMisClass/(double)testLab.rows()*100.0 << "%" << endl;
    cout << "--------------------------------------------\n\n";

	return 0;
}

