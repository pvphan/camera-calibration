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

- What were the innovations of Zhang calibration over the prior state of the art?

    - Previous calibration techniques required more expensive or procedures: specially made 3D calibration targets, or targets that are moved in a precise way.
    - Zhang's method requires only a 2D planar target (cheap to print) and requires no special movements

- Under what conditions will this calibration method fail?

    - If the calibration target undergoes pure, unknown translation, Zhang's method will not work.
    - This is because additional views on the same model plane do not add additional constraints.
    - But if the translation of the target is precisely known, then calibration is possible if we impose those constraints.

- At a high level, what are the steps to the Zhang calibration algorithm?

    - Collect feature points (2D / 3D point associations) from several images (assumed to be done)
    - Estimate the intrinsic and extrinsic parameters using the closed form solution
    - Estimate the radial distortion parameters
    - Refine all parameters by minimizing

- What is SVD, DLT, and QR, and how do they relate to Zhang calibration?

    - In the DLT case, QR-decomposition is used to decouple the intrinsics (K) and the rotation matrix (R) from the full projection matrix P.
        - But in Zhang's method, we cannot use QR-decomposition because the product contains the intrinsic matrix and a matrix which is not orthogonal (r1, r2, t)
            - x = P * X, P = [H | h], H = K * R
            - QR-decomposition separates H into its two products: an orthogonal matrix (the rotation matrix, R) and an upper-diagonal matrix (the intrinsic matrix, K)
        - Instead, we will drop the z terms (every 3rd col) of the linear system and solve for the 3x3 homography H

    - Still need to estimate K from H: H = K * [r1 r2 t]. So we need a custom solution to exploit properties we know about K, r1, and r2
        1. Exploit constraints on K, r1, r2
        2. Define a matrix B = K^-T * K^-1
        3. Compute B by solving another homogeneous linear system
        4. Decompose matrix B to get K

