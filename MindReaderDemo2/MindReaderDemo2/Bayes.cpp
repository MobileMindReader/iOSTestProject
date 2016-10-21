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

using namespace std;

struct MODEL {
    int dimensions;// = 1;
    double alpha;// = 2.0f;
    float noiseMean;// = 0.0f;
    
    float sigma;// = 0.2f;
    float beta;// = 25.0f; // 1 / sigma^2
};


// Basis function stuff
Mat gaussBaseFunc(float mu, float spatial, CvMat x);
static const int numBasisFuncs = 9;
float gaussBaseMeans[numBasisFuncs] = {
    2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2
};
float gaussBaseSpatials[numBasisFuncs] = {
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
};

// Samples
int numSamples = 1e5;

tuple<double, double> Bayes::bayes() {
    
    struct MODEL model;
    model.dimensions = 1;
    model.alpha = 0.3;
    model.sigma = 0.2;
    model.beta = 1/powf(model.sigma, 2);
    model.noiseMean = 0.0;
    
    
    Mat (*functions[numBasisFuncs]) (float mu, float spatial, CvMat x);
    for (int i = 0; i < numBasisFuncs; ++i) {
        functions[i] = gaussBaseFunc;
    }
    
    timeval tv;
    gettimeofday(&tv, 0);
    int randSeed = rand()*int(tv.tv_usec);
    CvRNG rng_state = cvRNG(randSeed);

    CvMat *wTemp = cvCreateMat(numBasisFuncs+1, 1, CV_64F);
    CvMat w;        //    cv::Mat w2(10, 1, CV_64F);
    cvGetRows(wTemp, &w, 0, numBasisFuncs+1);
    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(1/model.alpha));

//    Scalar means, stds;
//    meanStdDev(Mat(&w), means, stds);
//    cout << stds << endl;
    
//    float some[10] = {-0.71950376, 0.7439397, 0.044576742, 0.09926828, -0.028395038, 0.24359779, 0.44448805, 0.28344581, -0.50028253, -0.39646003};
//    Mat w = cv::Mat(numBasisFuncs+1, 1, CV_64F, some);
    
//    cout << cv::Mat(&w).t() << endl;
//    cout << w.cols << endl;
    
    
    // Generate samples
    CvMat *xTemp = cvCreateMat(1, numSamples, CV_64F);
    CvMat x;
    cvGetCols(xTemp, &x, 0, numSamples);
    cvRandArr( &rng_state, &x, CV_RAND_UNI, cvScalar(-2), cvScalar(2));

    
//    float trainX[200] = {-1.6891344, -0.81146395, -0.80328029, -0.39866108, -1.3465253, 0.097953826, -0.7979992, 1.07725, -1.0757492, -1.7827187, -0.69529963, 1.7106341, 0.066623293, -0.054572485, -1.4552741, 0.92592323, 0.64163762, 1.2565767, 0.026520684, 1.2557539, 0.39555514, 1.8845409, -1.409718, 1.728619, -1.0571704, 1.5618118, -0.0048333192, 0.44179261, 0.74815977, -0.87224507, -0.089025378, 0.26626098, 0.51490951, -1.8638452, -0.9460777, 0.71370357, 1.3420378, -1.7174065, 1.2482944, 0.88903618, -1.6751707, 0.5978539, 0.53131962, 0.91448861, 1.1754308, -1.6103787, -0.74267596, 1.1051271, -0.42703152, 1.5056521, -0.21128212, -1.0872496, 1.4924588, 0.42427298, 0.27538982, 0.53397572, -1.504577, -1.5211203, -0.1970688, 0.58465552, -1.0898799, -1.0193743, 0.84911251, 1.79223, 1.1148257, 0.65589887, 1.9717593, -0.16896585, 1.5360786, -0.92213744, -0.84100235, 0.51708406, 0.8543551, -1.1667957, 1.6228566, -1.1221099, -0.78108072, -0.73080474, 1.1294609, 0.45708948, 1.2431792, -0.39059857, -1.6147022, 1.5833527, 0.11285317, -0.14612544, 1.2216982, 1.7475219, -1.4405506, -0.13084558, -1.7638396, 1.2828581, 0.69804454, 0.75347793, -1.2873015, 0.47282866, -1.8089161, -0.3372989, -1.5656923, 1.0041701, -1.5486854, 0.027653014, -0.37790063, 0.27964497, 1.7282027, 1.6121138, 1.7615677, -1.9180593, -1.740833, 0.2058917, 1.4255201, -0.5218845, -1.7778432, 0.81647366, 1.481274, -1.1702905, -1.6530917, 0.45649746, -0.42804277, -0.11032233, 0.27182037, 0.91895777, 0.40872532, -1.8344363, -1.8409078, 1.3648391, 1.2846701, -1.9303113, 0.14925098, 0.56170356, -1.5165149, 1.6153481, 0.22302708, -0.12705375, 0.82200193, 1.4090627, -0.29597247, 0.11656475, -0.57634574, 0.49477795, 0.85218865, -0.28047559, 0.83837241, 1.7273241, -1.7263123, -1.776225, -1.9977804, -0.92418128, -0.3406437, -0.15862052, 1.1564885, 1.5996418, -1.0177807, -0.57910794, -1.0526571, 1.4335953, -0.82716089, 0.54188311, 1.6914917, -0.73987627, -0.71333969, -1.7712414, 0.024425674, -1.1080514, 1.7922405, 1.1075817, -0.97736031, 0.56951803, 1.2147849, -1.7216859, -1.8620257, -0.96800804, -0.33747026, -1.4624453, 1.1017243, -1.008311, -1.4931763, -0.45205718, 1.7426289, -0.34963641, -0.16979393, -0.5696826, -0.75587124, -1.7632818, -1.0326688, -1.768737, -1.5099456, -1.894125, -0.95073771, -1.4433213, 0.78398508, 1.5944117, 0.15183277, -1.1915636, -1.168054, -1.1269232, 0.21778987, -1.387993, 0.40843394, 1.1482033};
//    Mat x = Mat(1, numSamples, CV_64F, trainX);
    
    
    Mat y = phi(functions, gaussBaseMeans, gaussBaseSpatials, w, x);
    Mat targets = Mat(1, numSamples, CV_64F);
    
    // Random target noise
    CvMat *targetNoise = cvCreateMat(1, numSamples, CV_64F);
    cvRandArr( &rng_state, targetNoise, CV_RAND_NORMAL, cvScalar(model.noiseMean), cvScalar(model.sigma));
    
//    float targetNoises[200] = {0.54187697, -0.23723364, 0.04156081, -0.15157653, 0.07778772, -0.58498847, 0.33338091, -0.10090246, -0.12051326, -0.1019985, -0.022633998, -0.30929959, -0.26692221, -0.015649071, -0.013466291, -0.10548122, -0.085448392, 0.084335722, -0.10230301, 0.29436901, -0.20315717, -0.094950698, 0.43172058, -0.3087551, -0.15090333, 0.011518011, 0.12109764, 0.074880265, -0.068537481, -0.44062421, 0.0097377654, -0.01061864, -0.15291207, -0.22583902, 0.21703704, -0.21326344, -0.084162459, 0.32184672, 0.22912657, 0.096540011, 0.066717193, 0.090986356, 0.13672248, 0.37862331, -0.21990736, -0.003352683, 0.0078201182, -0.16969179, -0.09087982, 0.13722709, 0.10446312, 0.18861113, 0.11816669, -0.31496853, 0.093197837, -0.034981105, 0.14753382, 0.1676091, 0.24194045, -0.21914013, 0.12961791, 0.16658919, -0.1511779, 0.22500888, 0.10089689, 0.21131623, 0.026996905, -0.097686686, 0.037190378, -0.073381647, -0.0092664091, 0.27759978, 0.061381377, 0.24852709, -0.05153675, -0.15800579, 0.094000243, 0.10220224, -0.0065002958, -0.40737697, 0.086745627, 0.019565482, 0.12402362, -0.080251902, 0.10181405, 0.37907016, -0.086863719, -0.1080996, -0.06580653, -0.18453877, -0.10160305, -0.21984629, 0.049436215, -0.067795888, -0.012287862, 0.084889643, -0.0065334118, -0.16628215, -0.030922998, -0.17403732, -0.22113541, -0.23924522, 0.20716488, 0.0267506, 0.12865981, 0.21712127, 0.22650097, 0.12647034, 0.026337758, 0.19145098, 0.049436036, -0.052483529, -0.19525975, 0.021692665, 0.29838639, -0.13657832, -0.036121648, 0.14689331, 0.097598635, 0.13922572, -0.097543225, -0.12861417, -0.14357492, 0.029159999, 0.033828478, -0.00011314197, -0.20456381, 0.010219778, 0.15246443, -0.17709404, 0.12014958, 0.24400194, -0.028969005, -0.15025115, 0.2273474, -0.26966733, -0.06920103, 0.12513809, 0.088366143, -0.10552964, 0.13418703, 0.25561538, -0.10445126, 0.15276386, 0.12422861, 0.18541937, -0.53717941, -0.29579937, -0.12223955, -0.090703048, -0.2429494, 0.067362785, 0.085739851, 0.12837212, -0.19540314, 0.14556222, -0.15017989, 0.30977428, 0.22266367, -0.13237828, 0.15247861, -0.09857399, 0.0071269842, -0.1113783, 0.13710944, -0.13239683, -0.26972246, -0.34686261, -0.17881523, -0.29179287, 0.15919122, 0.091584548, 0.029055566, 0.038665291, -0.021990886, 0.0222778, 0.37275332, -0.058316868, 0.29078287, -0.21355052, -0.1603611, -0.035892133, -0.0061936211, -0.22877324, -0.027462646, 0.22140172, 0.26185712, 0.0044532982, 0.114456, -0.27435026, -0.016796257, 0.1125555, 0.059728649, -0.13206245, -0.11362293, 0.088931844, -0.17602427, 0.3471069, 0.3194229, 0.18961906};
//    Mat targetNoise = Mat(1, numSamples, CV_64F, targetNoises);
    
    
//    Mat(1, numSamples, CV_64F, Scalar(0.0));
    add(y, Mat(targetNoise), targets);
    
//    cout << Mat(x) << endl;
//    cout << Mat(&w).t() << endl;
//    cout << targets << endl;
    
    /// Phi
    Mat Phi = PhiMatrix(functions, gaussBaseMeans, gaussBaseSpatials, x);
    Mat PhiT = Phi.t();
    
    
    Mat PhiTPhi = PhiT*Phi;
    Mat PhiTtargets = PhiT * targets.t();
    
    
//    Mat wML = Mat();
//    solve(PhiTPhi, PhiTtargets, wML);
    
    
    auto evidence = evidenceMaximisation(Phi, targets);
    
    cout << "True beta: " << model.beta << endl;
    cout << "Estimated: " << evidence.beta << endl;
    
    
    //// THIS IS IRRELEVANT AS sigma^2 = 1/beta
    //// Sigma
    Mat SNInv = evidence.alpha*Mat::eye(numBasisFuncs+1, numBasisFuncs+1, CV_64F) + (evidence.beta * PhiTPhi);
    Mat temp;
    solve(SNInv, PhiT, temp);
//    Mat sigmaSq = 1/beta + (Phi*temp);
//    double sigma = sqrt(mean(sigmaSq.diag())[0]);
//    cout << "True sigma: " << model.sigma << endl;
//    cout << "Estimated : " << sigma << endl;
    
    Mat mN = evidence.beta * (temp * targets.t());
    
//    for (int y = 0; y < w.rows; ++y) {
//        printf("%.4f ", Mat(&w).at<double>(y));
//    }
//    cout << endl;
//    for (int y = 0; y < w.rows; ++y) {
//        printf("%.4f ", wML.at<double>(y));
//    }
//    cout << endl;
//    for (int y = 0; y < w.rows; ++y) {
//        printf("%.4f ", mN.at<double>(y));
//    }
//    cout << endl;
    
    cout << numSamples << endl;
    
    return {model.alpha/model.beta, evidence.alpha / evidence.beta};
}


ModelEvidence Bayes::evidenceMaximisation(Mat Phi, Mat t) {
    
    double alpha = rand();      // Random initialize
    double beta = rand();       // Random initialize
    
    int M = numBasisFuncs+1;
    int N = numSamples;
    
    int maxIterations = 200;
    double tolerance = 1e-4;
    
    ModelEvidence evidence;
    
    Mat PhiT = Phi.t();
    Mat PhiTPhi = PhiT*Phi;
    
    Mat PhiTPhiEig = Mat();
    eigen(PhiTPhi, PhiTPhiEig);
    
    Mat llh = Mat::zeros(1, maxIterations, CV_64F);
    
    for (int i = 1; i < maxIterations; ++i) {
        Mat betaPhiTPhi =  beta*PhiTPhi;
        Mat A = alpha*Mat::eye(M, M, CV_64F) + betaPhiTPhi;
        Mat mNTemp = Mat();
        solve(A, PhiT, mNTemp);
        Mat mN = beta * (mNTemp * t.t());
        
        Mat lambda = beta * PhiTPhiEig;
        
        double gamma = 0;
        for (int j = 0; j < M; j++) {
            gamma += lambda.at<double>(j) /(alpha+lambda.at<double>(j));
        }
        alpha = gamma/(mN.dot(mN));
        
        Mat EwTemp = Mat();
        pow((t.t()-(Phi*mN)), 2, EwTemp);
        double Ew = sum(EwTemp)[0];
        
//        double betaInv = (1/(N-gamma)) * Ew;
        beta = 1/((1/(N-gamma)) * Ew);
        
        double Em = beta/2 * Ew + alpha/2 * mN.dot(mN);
        
        llh.at<double>(i) = 0.5*(M*log(alpha) + N*log(beta) - 2*Em - log(determinant(A)) - N*log(2*CV_PI));  /// Marginal log likelihood (3.86)
        if ( abs(llh.at<double>(i) - llh.at<double>(i-1)) < tolerance*abs(llh.at<double>(i-1)) )  {
            evidence.llh = llh.at<double>(i);
            cout << "Converged at iteration: " << i << ", " << llh.at<double>(i) << endl;
            break;
        }
    }
    
    evidence.alpha = alpha;
    evidence.beta = beta;
    
    return evidence;
}


Mat Bayes::PhiMatrix(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat x) {
    // x.rows = dimension = 1
    
    Mat Phi = Mat::ones(x.cols, numBasisFuncs+1, CV_64F);
    
    for (int j = 0; j < numBasisFuncs; ++j) {
        Mat temp = (*functions[j])(means[j], spatials[j], x);
        temp.copyTo(Phi.col(j+1));
    }
    
    return Phi;
}


Mat Bayes::phi(Mat (*functions[]) (float mu, float spatial, CvMat x), float means[], float spatials[], CvMat w, CvMat x) {
    // x.rows = dimension = 1
    
    Mat y = w.data.db[0] * Mat::ones(x.rows, x.cols, CV_64F);
    
    for (int j = 0; j < numBasisFuncs; ++j) {
        // The weight from w must be matching to basefunc
        Mat weightedFuncOutput = w.data.db[j+1] * (*functions[j])(means[j], spatials[j], x); //baseFunc(x);
        add(y, weightedFuncOutput.t(), y);
    }
    return y;
}


Mat gaussBaseFunc(float mu, float spatial, CvMat x) {
    Mat y = Mat(x.cols, 1, CV_64F);
    Mat xMeanSq = Mat(x.cols, 1, CV_64F);
    
    Mat xMean = Mat(&x) - double(mu);
    pow(xMean, 2, xMeanSq);
    
    Mat temp = -xMeanSq / double(2.0*powf(spatial, 2));
    exp(temp, y);
    
    return y.t();
}




void Bayes::testPerf() {
    
    int size = 1000;
    
//    timeval tv;
//    gettimeofday(&tv, 0);
//    int randSeed = rand()*int(tv.tv_usec);
//    CvRNG rng_state = cvRNG(randSeed);
    
    CvMat *wTemp = cvCreateMat(size, size, CV_64F);
    
    CvMat w;
    
//    cvGetMat(wTemp, &w);
    cvGetRows(wTemp, &w, 0, size);

//    cvRandArr( &rng_state, &w, CV_RAND_NORMAL, cvScalar(0.0), cvScalar(1.5));
    
    Mat A = Mat(size, size, CV_64F);
    
    randn(A, Scalar(0.0), Scalar(5));
    
    Mat B = A.inv();
    
    cout << B.row(10) << endl;
}


void mlstuff() {
    //    Mat wML = PhiTPhiInv * PhiTtargets;
    
    //    cout << "Real     : " << Mat(&w).t() << endl;
    //    cout << "Estimated: " << wML.t() << endl;
    //// Beta ML
    
    //    double invBeta = 0;
    //    for (int i = 0; i < numSamples; ++i) {
    //        invBeta += pow(targets.at<double>(i) - wML.dot(Phi.row(i).t()), 2);
    //    }
    //    double betaML = 1/(invBeta/numSamples);

}


//void Bayes::doStuff() {
//    
//    
//    cv::Mat C = (cv::Mat_<double>(2,2) << 0.1, 0.2, 0.3, 0.4);
//    cout << "C = " << endl << " " << C << endl << endl;
//    
//    
//    cv::Mat means(1, 1, CV_64F, cv::Scalar(0,0,0));
//    cv::Mat stds(1, 1, CV_64F, cv::Scalar(0.2,0.2,0.2));
//    
//    cv::Mat output(1, 1, CV_64F);
//    
//    cv::randn(output, means, stds);
//    
//    cv::Mat some = cv::Mat::zeros(10, 10, CV_64F); //(10,10,0);
//    cv::Mat idx = cv::Mat::eye(10, 10, CV_64F);
//    
//    cout << "Output random variable" << endl;
//    cout << output << endl;
//    
//    cv::Mat yolo;
//    yolo = (cv::Mat_<float>(2,2) << 0.1,0.2,0.3,0.4);
//    
//    cout << "YOLO" << endl << yolo << endl;
//}

//void Bayes::test() {
//    const int K = 10;
//    int i, j, k, accuracy;
//    float response;
//    int train_sample_count = 100;
//    CvRNG rng_state = cvRNG(-1);
//    CvMat* trainData = cvCreateMat( train_sample_count, 2, CV_64FC1 );
//    CvMat* trainClasses = cvCreateMat( train_sample_count, 1, CV_64FC1 );
//    IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
//    float _sample[2];
//    CvMat sample = cvMat( 1, 2, CV_64FC1, _sample );
//    cvZero( img );
//    
//    CvMat trainData1, trainData2, trainClasses1, trainClasses2;
//    
//    // form the training samples
//    cvGetRows( trainData, &trainData1, 0, train_sample_count/2 );
//    cvRandArr( &rng_state, &trainData1, CV_RAND_NORMAL, cvScalar(200,200), cvScalar(50,50) );
//    
//    cout << trainData1.cols << endl;
//}

