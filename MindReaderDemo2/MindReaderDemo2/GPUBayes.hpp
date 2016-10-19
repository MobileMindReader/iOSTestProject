//
//  GPUBayes.hpp
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 19/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#ifndef GPUBayes_hpp
#define GPUBayes_hpp

#include <stdio.h>

#include <opencv2/core/types_c.h>
#include <opencv2/core/gpumat.hpp>

using namespace cv;

class GPUBayes {
private:
    
    Mat phi(CvMat w, CvMat x);
    Mat baseFunc(CvMat x);
    Mat PhiMatrix(CvMat x);
    float baseFunc1(float x);
    
public:
    void bayes();
    void testGpu();
};


#endif /* GPUBayes_hpp */
