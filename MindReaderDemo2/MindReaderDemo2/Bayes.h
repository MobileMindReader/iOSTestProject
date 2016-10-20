//
//  Something.h
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 13/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#ifndef Bayes_h
#define Bayes_h

#include <opencv2/core/types_c.h>
#include <opencv2/core/gpumat.hpp>

using namespace cv;

class Bayes {
private:
    
    
    Mat phi(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat w, CvMat x);
    Mat PhiMatrix(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat x);
    
    std::tuple<double, double> alphaBetaEstimation(Mat Phi, Mat t);
public:
    
    void doStuff();
    void bayes();
    void test();
    void testPerf();
};


#endif /* Something_h */
