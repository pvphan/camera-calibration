# Camera Calibration using Zhang Method

Current status: WIP

Usage: TBD


## Goals:

### Write Zhang calibration by hand

Create a simple interface for calibrating camera intrinsics from a text file of 2D to 3D point correspondences.
Don't use OpenCV, instead code it by hand.


### Answer the following questions

- What were the innovations of Zhang calibration over the prior state of the art?

    - Previous calibration techniques required more expensive or procedures: specially made 3D calibration targets, or targets that are moved in a precise way.
    - Zhang's method requires only a 2D planar target (cheap to print) and requires no special movements

- Under what conditions will this calibration method fail?

    - If the calibration target undergoes pure, unknown translation, Zhang's method will not work.
    - But if the translation of the target is precisely known, then calibration is possible if we impose those constraints.

- At a high level, what are the steps to the Zhang calibration algorithm?

    - Collect feature points (2D / 3D point associations) from several images (assumed to be done)
    - Estimate the intrinsic and extrinsic parameters using the closed form solution
    - Estimate the radial distortion parameters
    - Refine all parameters by minimizing

- What is SVD, DLT, and QR, and how do they relate to Zhang calibration?

    - QR-decomposition is used to decouple the intrinsics (K) and the rotation matrix (R) from the full projection matrix P
    - x = P * X, P = [H | h], H = K * R
    - QR-decomposition separates H into its two products: an orthogonal matrix (the rotation matrix, R) and an upper-diagonal matrix (the intrinsic matrix, K)


## Notes:
- Need 6 points to find transform matrix P in the equation x = P * X. 11 unknowns, each point gives 2 variables, so 11 / 2 = 5.5 ~= 6
- P = [H | h]
- X0 = -H^-1 * h
- To decompose H = KR into the intrinsic and rotation matrices, use QR-decomposition.
    - In QR-decomposition, Q is a rotation matrix, R is a triangular matrix.
    - H^-1 = (K * R)^-1 = R^-1 * K^-1 = R^T * K^-1
        - Q = R^T
        - R = K^-1
    - Need to normalize K, e.g. K = 1/K33 * K
    - Need to do a coordinate tranform by a rotation of 180 deg
        - K = K * R(z, 180)
        - R = R(z, 180) * R

- DLT in a nutshell
    1. Build M for the linear system: M is (2 * i, 12) and p is (12, 1). M * p = 0.
        For every point we measure, we add 2 rows to the matrix M (minimum of 6 points which is 12 rows).

    2. Solve by SVD M = U S V^T, solution is the last column of V, which are the values of p which gives us P.
    3. Solve for K, R, X0. Let P = [H | h]
        - X0 = -H^-1 * h
        - QR(H^-1) = R^T * K^-1
        - R = R(z, 180) * R
        - K = (1/K33) * K * R(z, 180)

- numpy has SVD and QR


## References:
- (paper) [Zhengyou Zhang: A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
- (paper) [Wilhelm Burger: Zhang's Camera Calibration Algorithm: In-Depth Tutorial and Implementation](https://www.researchgate.net/publication/303233579_Zhang's_Camera_Calibration_Algorithm_In-Depth_Tutorial_and_Implementation).
- (video) [Cyrill Stachniss: Direct Linear Transform](https://www.youtube.com/watch?v=3NcQbZu6xt8&ab_channel=CyrillStachniss)
- (video) [Cyrill Stachniss: Camera Calibration using Zhang's Method](https://www.youtube.com/watch?v=-9He7Nu3u8s&ab_channel=CyrillStachniss)

