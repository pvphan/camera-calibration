# Camera Calibration using Zhang's Method

[![Build Status](https://app.travis-ci.com/pvphan/camera-calibration.svg?branch=main)](https://app.travis-ci.com/pvphan/camera-calibration)

A simple library for calibrating camera intrinsics from sensor (2D) and model point (3D) correspondences.
Written with few external dependencies (numpy, sympy, imageio) for a deeper understanding.
Also generates synthetic datasets for testing and rudimentary visualization.

Prerequisites: `make`, `docker`


## TODO:

- [ ] Vectorize the Jacobian computation (takes ~14 sec per iteration of Levenberg-Marquardt currently)
- [ ] Button up as python package, add instructions to README
- [ ] Support fisheye distortion model


## References:
- (paper) [Wilhelm Burger: Zhang's Camera Calibration Algorithm: In-Depth Tutorial and Implementation](https://www.researchgate.net/publication/303233579_Zhang's_Camera_Calibration_Algorithm_In-Depth_Tutorial_and_Implementation).
- (paper) [Zhengyou Zhang: A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
- (video) [Cyrill Stachniss: Direct Linear Transform](https://www.youtube.com/watch?v=3NcQbZu6xt8&ab_channel=CyrillStachniss)
- (video) [Cyrill Stachniss: Camera Calibration using Zhang's Method](https://www.youtube.com/watch?v=-9He7Nu3u8s&ab_channel=CyrillStachniss)
