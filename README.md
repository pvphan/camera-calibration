# Camera Calibration using Zhang Method

Current status: WIP
Usage: TBD

This code is based on the tutorial written by Wilhelm Burger on Zhengyou Zhang's calibration method:
[Zhang's Camera Calibration Algorithm: In-Depth Tutorial and Implementation](https://www.researchgate.net/publication/303233579_Zhang's_Camera_Calibration_Algorithm_In-Depth_Tutorial_and_Implementation)


Some helpful videos:
- [Prof. Cyrill Stachniss: Direct Linear Transform](https://www.youtube.com/watch?v=3NcQbZu6xt8&ab_channel=CyrillStachniss)


Notes:
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

- DLT  in a nutshell
    1. Build M for the linear system: M is (2 * i, 12) and p is (12, 1). M * p = 0.
        For every point we measure, we add 2 rows to the matrix M (minimum of 6 points which is 12 rows).

    2. Solve by SVD M = U S V^T, solution is the last column of V, which are the values of p which gives us P.
    3. Solve for K, R, X0. Let P = [H | h]
        - X0 = -H^-1 * h
        - QR(H^-1) = R^T * K^-1
        - R = R(z, 180) * R
        - K = (1/K33) * K * R(z, 180)

- numpy has SVD and QR
