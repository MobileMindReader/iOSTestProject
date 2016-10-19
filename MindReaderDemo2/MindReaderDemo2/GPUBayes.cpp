//
//  GPUBayes.cpp
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 19/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#include "GPUBayes.hpp"
#include <iostream>
#include <sys/time.h>


struct MODEL {
    float alpha = 0.28;
    float noiseMean = 0;
    float sigma = 0.2;
    float beta = 25;
    int dimensions = 1;
};

using namespace gpu;

void GPUBayes::testGpu() {
//    int size = 1000;
//    CvRNG rng_state = cvRNG(-1);
//    CvMat *wTemp = cvCreateMat(size, size, CV_32F);
//    
//    CvMat w;
//    
//    cvGetMat(wTemp, &w, 0, size);
//    
//    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(0.5));
//    
//    Mat test = Mat(&w);
//    GpuMat testGpu = GpuMat(&w);
    
}


void GPUBayes::bayes() {
    
    struct MODEL model;
    
    int numBasisFuncs = 2;
    
    timeval tv;
    gettimeofday(&tv, 0);
    int randSeed = rand()*int(tv.tv_usec);
    
    
    CvRNG rng_state = cvRNG(randSeed);
    CvMat *wTemp = cvCreateMat(numBasisFuncs, 1, CV_32F);
    
    CvMat w;        //    cv::Mat w2(10, 1, CV_32F);
    
    cvGetRows(wTemp, &w, 0, numBasisFuncs);
    
    
    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(0.5));
    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(0.5));
    
    
    // Generate samples
    int numSamples = 200;
    CvMat *xTemp = cvCreateMat(1, numSamples, CV_32F);
    CvMat x;
    cvGetCols(xTemp, &x, 0, numSamples);
    cvRandArr( &rng_state, &x, CV_RAND_UNI, cvScalar(-2), cvScalar(2));
    //    for (int i = 0; i < numSamples; ++i) {
    //        std::cout << x.data.fl[i] << std::endl;
    //    }
    
    //    CvMat y = phi(w, x);
    Mat y = phi(w, x);
    Mat targets = Mat(1, numSamples, CV_32F);
    
    // Random target noise
    CvMat *targetNoise = cvCreateMat(1, numSamples, CV_32F);
    cvRandArr( &rng_state, targetNoise, CV_RAND_NORMAL, cvScalar(0), cvScalar(0.2));
    
    add(y, Mat(targetNoise), targets);
    
    /// Phi
    Mat Phi = PhiMatrix(x);
    Mat PhiT = Phi.t();
    
    
    Mat PhiTPhi = PhiT*Phi;
    Mat PhiTPhiInv = PhiTPhi.inv();
    Mat PhiTtargets = PhiT * targets.t();
    Mat wML = PhiTPhiInv * PhiTtargets;
    
    std::cout << "Real     : " << Mat(&w).rowRange(0, 2).t() << std::endl;
    std::cout << "Estimated: " << wML.t() << std::endl;
    
    
    //// Beta ML
    
    float invBeta = 0;
    for (int i = 0; i < numSamples; ++i) {
        invBeta += powf(targets.at<float>(i) - wML.dot(Phi.row(i).t()), 2);
    }
    float betaML = 1/(invBeta/numSamples);
    
    std::cout << "True beta: " << model.beta << std::endl;
    std::cout << "Estimated: " << betaML << std::endl;
    
    /// ALPHA AND BETA Estimation done with evidence function stuff
    
    float beta = betaML;
    
    
    
    //// Sigma
    Mat SNInv = model.alpha*Mat::eye(numBasisFuncs, numBasisFuncs, CV_32F) + (beta * (PhiT * Phi));
    Mat sigmaSq = 1/beta + Phi*(SNInv.inv()*PhiT);
    float sigma = sqrt(mean(sigmaSq.diag())[0]);
    
    std::cout << "True sigma: " << model.sigma << std::endl;
    std::cout << "Estimated : " << sigma << std::endl;
    
    
    
    
}

Mat GPUBayes::PhiMatrix(CvMat x) {
    int numBasisFuncs = 1;
    Mat Phi = Mat::ones(x.cols, numBasisFuncs+1, CV_32F);
    
    for (int i = 1; i <= numBasisFuncs; i++) {
        Mat temp = baseFunc(x);
        temp.col(0).copyTo(Phi.col(i));
    }
    
    return Phi;
}

Mat GPUBayes::phi(CvMat w, CvMat x) {
    int numBaseFuncs = 1;   // x.rows = dimension = 1
    
    Mat yTemp = w.data.fl[0] * Mat::ones(x.rows, x.cols, CV_32F);
    Mat y = Mat(x.rows, x.cols, CV_32F);
    
    for (int j = 0; j < numBaseFuncs; ++j) {
        // The weight from w must be matching to basefunc
        //            yTemp.at<float>(1,i) += w.data.fl[1] * baseFunc1(x.data.fl[i]);
        Mat weightedFuncOutput = w.data.fl[1] * baseFunc(x);
        add(yTemp, weightedFuncOutput.t(), y);
    }
    return y;
}

cv::Mat GPUBayes::baseFunc(CvMat x) {
    cv::Mat some = cv::Mat::ones(x.cols, 1, CV_32F);
    
    //    std::cout << cv::Mat(&x) << std::endl;
    
    float spatial = 0.5;
    float mu = 0.0;
    //    cvPow(<#const CvArr *src#>, <#CvArr *dst#>, <#double power#>)
    
    for (int i = 0; i < x.cols; ++i) {
        float temp = expf(-powf(x.data.fl[i]-mu, 2)/(powf(2.0*spatial, 2)));
        some.at<float>(i,0) = temp;
    }
    
    return some;
}

float GPUBayes::baseFunc1(float x) {
    float spatial = 0.5;
    float mu = 0.0;
    
    float y  = expf(-powf(x-mu, 2)/(powf(2.0*spatial, 2)));
    return y;
}
