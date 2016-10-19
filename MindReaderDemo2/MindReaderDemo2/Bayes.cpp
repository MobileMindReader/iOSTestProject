//
//  Bayes.cpp
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 13/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#include "Bayes.h"
#include <iostream>
#include <sys/time.h>
//#include <opencv2/nonfree/gpu.hpp>

struct MODEL {
    float alpha = 0.28;
    float noiseMean = 0;
    float sigma = 0.2;
    float beta = 25;
    int dimensions = 1;
};

Mat gaussBaseFunc(float mu, float spatial, CvMat x);
static const int numBasisFuncs = 9;

void Bayes::doStuff() {

    
    cv::Mat C = (cv::Mat_<double>(2,2) << 0.1, 0.2, 0.3, 0.4);
    std::cout << "C = " << std::endl << " " << C << std::endl << std::endl;
    
    
    cv::Mat means(1, 1, CV_32F, cv::Scalar(0,0,0));
    cv::Mat stds(1, 1, CV_32F, cv::Scalar(0.2,0.2,0.2));
    
    cv::Mat output(1, 1, CV_32F);
    
    cv::randn(output, means, stds);
    
    cv::Mat some = cv::Mat::zeros(10, 10, CV_32F); //(10,10,0);
    cv::Mat idx = cv::Mat::eye(10, 10, CV_32F);
    
    std::cout << "Output random variable" << std::endl;
    std::cout << output << std::endl;
    
    cv::Mat yolo;
    yolo = (cv::Mat_<float>(2,2) << 0.1,0.2,0.3,0.4);
    
    std::cout << "YOLO" << std::endl << yolo << std::endl;
}


void Bayes::bayes() {
    
    struct MODEL model;
    
    float gaussBaseMeans[numBasisFuncs] = {
        -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    };
    float gaussBaseSpatials[numBasisFuncs] = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    };
    Mat (*functions[numBasisFuncs]) (float mu, float spatial, CvMat x);
    for (int i = 0; i < numBasisFuncs; ++i) {
        functions[i] = gaussBaseFunc;
    }
    
    timeval tv;
    gettimeofday(&tv, 0);
    int randSeed = rand()*int(tv.tv_usec);
    
    CvRNG rng_state = cvRNG(randSeed);
    CvMat *wTemp = cvCreateMat(numBasisFuncs+1, 1, CV_32F);
    
    CvMat w;        //    cv::Mat w2(10, 1, CV_32F);
    cvGetRows(wTemp, &w, 0, numBasisFuncs+1);
    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(0.5));
    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(0.5));

//    std::cout << cv::Mat(&w).t() << std::endl;
//    cvRandArr(<#CvRNG *rng#>, <#CvArr *arr#>, <#int dist_type#>, <#CvScalar param1#>, <#CvScalar param2#>)
//    std::cout << w.cols << std::endl;
    
    
    // Generate samples
    int numSamples = 200;
    CvMat *xTemp = cvCreateMat(1, numSamples, CV_32F);
    CvMat x;
    cvGetCols(xTemp, &x, 0, numSamples);
    cvRandArr( &rng_state, &x, CV_RAND_UNI, cvScalar(-2), cvScalar(2));

    
//    Mat y = phi(w, x);
    Mat y = phi(functions, gaussBaseMeans, gaussBaseSpatials, w, x);
//    Mat yComp = phi(w, x);
//    std::cout << y << std::endl;
//    std::cout << yComp << std::endl;
    
    Mat targets = Mat(1, numSamples, CV_32F);
    
    // Random target noise
    CvMat *targetNoise = cvCreateMat(1, numSamples, CV_32F);
    cvRandArr( &rng_state, targetNoise, CV_RAND_NORMAL, cvScalar(0), cvScalar(0.2));
    
    add(y, Mat(targetNoise), targets);
    
//    cv::Scalar mean, stddev;
//    cv::meanStdDev(cv::Mat(&targetNoise), mean, stddev);
//    std::cout << cv::Mat(&targetNoise) << std::endl;
//    std::cout << mean << stddev << std::endl;
    
    
    /// Phi
//    Mat Phi = PhiMatrix(x);
    Mat Phi = PhiMatrix(functions, gaussBaseMeans, gaussBaseSpatials, x);
//    std::cout << Phi << std::endl;
    
    Mat PhiT = Phi.t();
    
    
    // TODO: Use different random seed methods in order to get more random
    
    Mat PhiTPhi = PhiT*Phi;
    Mat PhiTPhiInv = PhiTPhi.inv();
    Mat PhiTtargets = PhiT * targets.t();
    Mat wML = PhiTPhiInv * PhiTtargets;
    
    std::cout << "Real     : " << Mat(&w).t() << std::endl;
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
    Mat SNInv = model.alpha*Mat::eye(numBasisFuncs+1, numBasisFuncs+1, CV_32F) + (beta * (PhiT * Phi));
    Mat sigmaSq = 1/beta + Phi*(SNInv.inv()*PhiT);
    float sigma = sqrt(mean(sigmaSq.diag())[0]);

    std::cout << "True sigma: " << model.sigma << std::endl;
    std::cout << "Estimated : " << sigma << std::endl;
    
}

Mat Bayes::PhiMatrix(CvMat x) {
    Mat Phi = Mat::ones(x.cols, numBasisFuncs+1, CV_32F);
    
    for (int i = 1; i <= numBasisFuncs; i++) {
        Mat temp = baseFunc(x);
        temp.col(0).copyTo(Phi.col(i));
    }
    
    return Phi;
}

Mat Bayes::PhiMatrix(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat x) {
    // x.rows = dimension = 1
    
    Mat Phi = Mat::ones(x.cols, numBasisFuncs+1, CV_32F);
    
    for (int j = 0; j < numBasisFuncs; ++j) {
        Mat temp = (*functions[j])(means[j], spatials[j], x);
        temp.copyTo(Phi.col(j+1));
//        std::cout << Phi << std::endl;
    }
    
    return Phi;
}


Mat Bayes::phi(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat w, CvMat x) {
    // x.rows = dimension = 1
    
    Mat y = Mat(x.rows, x.cols, CV_32F);
    Mat yTemp = w.data.fl[0] * Mat::ones(x.rows, x.cols, CV_32F);
    
    for (int j = 0; j < numBasisFuncs; ++j) {
        // The weight from w must be matching to basefunc
        Mat weightedFuncOutput = w.data.fl[j+1] * (*functions[j])(means[j], spatials[j], x); //baseFunc(x);
        add(yTemp, weightedFuncOutput.t(), y);
    }
    return y;
}

Mat Bayes::phi(CvMat w, CvMat x) {
    // x.rows = dimension = 1
    
    Mat y = Mat(x.rows, x.cols, CV_32F);
    Mat yTemp = w.data.fl[0] * Mat::ones(x.rows, x.cols, CV_32F);
    for (int j = 0; j < numBasisFuncs; ++j) {
        // The weight from w must be matching to basefunc
        Mat weightedFuncOutput = w.data.fl[1] * baseFunc(x);
        add(yTemp, weightedFuncOutput.t(), y);
    }
    return y;
}

Mat gaussBaseFunc(float mu, float spatial, CvMat x) {
    Mat y = Mat(x.cols, 1, CV_32F);
    Mat xMeanSq = Mat(x.cols, 1, CV_32F);
    
    Mat xMean = Mat(&x) - mu;
    pow(xMean, 2, xMeanSq);
    
    Mat temp = -xMeanSq / powf(2.0*spatial, 2);
    exp(temp, y);
    
    return y.t();
}

Mat Bayes::baseFunc(CvMat x) {
    Mat y = Mat(x.cols, 1, CV_32F);
    Mat xMeanSq = Mat(x.cols, 1, CV_32F);
    
    float spatial = 0.5;
    float mu = -2.0;
    
    
    Mat xMean = Mat(&x) - mu;
    pow(xMean, 2, xMeanSq);
    
    Mat temp = -xMeanSq / powf(2.0*spatial, 2);
    exp(temp, y);
    
//    std::cout << y << std::endl;
    return y.t();
}

Mat Bayes::baseFuncOld(CvMat x) {
    Mat some = Mat::ones(x.cols, 1, CV_32F);
    float spatial = 0.5;
    float mu = -2.0;
    
    for (int i = 0; i < x.cols; ++i) {
        float temp = expf(-powf(x.data.fl[i]-mu, 2)/(powf(2.0*spatial, 2)));
        some.at<float>(i,0) = temp;
    }
    
    return some;
}

float Bayes::baseFunc1(float x) {
    float spatial = 0.5;
    float mu = 0.0;
    
    float y  = expf(-powf(x-mu, 2)/(powf(2.0*spatial, 2)));
    return y;
}


void Bayes::testPerf() {
    
    int size = 1000;
    
    timeval tv;
    gettimeofday(&tv, 0);
    int randSeed = rand()*int(tv.tv_usec);
    
    CvRNG rng_state = cvRNG(randSeed);
    CvMat *wTemp = cvCreateMat(size, size, CV_32F);
    
    CvMat w;
    
//    cvGetMat(wTemp, &w);
    cvGetRows(wTemp, &w, 0, size);

//    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(1.5));
    
    Mat A = Mat(size, size, CV_32F);
    
    randn(A, Scalar(0.0), Scalar(5));
    
    Mat B = A.inv();
    
    std::cout << B.row(10) << std::endl;
}


//void Bayes::test() {
//    const int K = 10;
//    int i, j, k, accuracy;
//    float response;
//    int train_sample_count = 100;
//    CvRNG rng_state = cvRNG(-1);
//    CvMat* trainData = cvCreateMat( train_sample_count, 2, CV_32FC1 );
//    CvMat* trainClasses = cvCreateMat( train_sample_count, 1, CV_32FC1 );
//    IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
//    float _sample[2];
//    CvMat sample = cvMat( 1, 2, CV_32FC1, _sample );
//    cvZero( img );
//    
//    CvMat trainData1, trainData2, trainClasses1, trainClasses2;
//    
//    // form the training samples
//    cvGetRows( trainData, &trainData1, 0, train_sample_count/2 );
//    cvRandArr( &rng_state, &trainData1, CV_RAND_NORMAL, cvScalar(200,200), cvScalar(50,50) );
//    
//    std::cout << trainData1.cols << std::endl;
//}

