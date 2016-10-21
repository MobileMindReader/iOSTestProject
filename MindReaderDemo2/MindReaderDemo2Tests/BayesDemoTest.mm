//
//  BayesDemoTest.m
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 18/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#import <XCTest/XCTest.h>
#import "ViewController.h"
#import "Bayes.h"

@interface BayesDemoTest : XCTestCase

@end

@implementation BayesDemoTest

//- (void)setUp {
//    [super setUp];
//    
//    // Put setup code here. This method is called before the invocation of each test method in the class.
//    
//    // In UI tests it is usually best to stop immediately when a failure occurs.
//    self.continueAfterFailure = NO;
//    // UI tests must launch the application that they test. Doing this in setup will make sure it happens for each test method.
//    [[[XCUIApplication alloc] init] launch];
//
//    // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
//}
//
//- (void)tearDown {
//    // Put teardown code here. This method is called after the invocation of each test method in the class.
//    [super tearDown];
//}


- (void)testBayes {
    
//    ViewController *controller = [[ViewController alloc] init];
//    [controller something];

    Bayes *bayes;
    
    double val = 0;
    double trueVal = 0;
    int iterations = 20;
    
    for (int i = 0; i < iterations; ++i) {
        auto ratio = bayes->bayes();
        NSLog(@"%.4f : %.4f", std::get<0>(ratio), std::get<1>(ratio));
        trueVal = std::get<0>(ratio);
        val += std::get<1>(ratio);
    }
    
    NSLog(@"####################################");
    
    NSLog(@"%.4f", val/iterations);
    
}

- (void)testTimePerf {
    Bayes *bayes;
//    
//    measureBlock:^{
//        
//    };
//    
    
    
    [self measureBlock:^{
        XCTestExpectation *expectation = [self expectationWithDescription:@""];
        
        
        auto ratio = bayes->bayes();
        
        [expectation fulfill];
        
        [self waitForExpectationsWithTimeout:10 handler:^(NSError *error) {
            XCTAssertNil(error);
        }];
    }];

}

- (void)testPerf {
    
    Bayes *bayes;
    
    //    Something some = Something();
    bayes->testPerf();
}


@end
