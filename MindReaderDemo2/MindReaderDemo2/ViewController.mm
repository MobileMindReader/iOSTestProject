//
//  ViewController.m
//  MindReaderDemo2
//
//  Created by Emil Maagaard on 13/10/2016.
//  Copyright © 2016 MaagaardApps. All rights reserved.
//

#import "ViewController.h"

#import "Bayes.h"


@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
//    [self something];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (void)something {
    
    NSLog(@"Do Bayes");
    
//    cv::Mat some = cvMat(10, 10, 0, 0);
    Bayes *bayes;
//    Something some = Something();
    bayes->bayes(1e3);
    
    
}

@end
