import matplotlib.pyplot as plt
import numpy as np
from   scipy import fftpack
from   scipy.linalg import qr, norm
from   PIL import Image
import time
import math as math
import random
from IPython.display import display, HTML
import imageio
import sys
import scipy


def adaptive_rank_determination(M,epsilon,blocking=16,iteration_max=10):
    """
    Adaptive rank determination based on randomization
    We look for an orthogonal matrix Q such that 
    $\|A - Q Q^T A \|_F \le \epsilon \|A \|_F$
    where $\|.\|_F$ denotes the Frobenius norm.
    
    Reference: Adapted from Section 12 of P.G. Martinsson, "Randomized methods for matrix computations", 2019, 
    ArXiv 1607.01649 [https://arxiv.org/pdf/1607.01649.pdfs]. The original algorithm corresponds to 
    Figure 12. 
    
    Input: 
    A:        matrix to be analysed [array type] (of shape (m,n))
    epsilon:  relative threshold related to the accuracy of the projection (in the Frobenius norm) (0<= epsilon <= 1)
    blocking: blocking parameter to be used in the implementation (for efficiency reasons) [int]
    
    Output:
    Q: matrix with orthonormal columns such that $ \|A - Q Q^T A \|_F \le \epsilon \|A \|_F$
    erank: estimated rank (upper bound of epsilon-rank(A)), erank is here a multiple of min(blocking,n). 
    ratio: current ratio of $\|A - Q Q^T A \|_F4$ divided by $\|A \|_F$
    iteration: nombre d'itérations réalisé au sein de l'algorithme
    """ 
    A             = M.copy()
    m, n          = A.shape[:]
    # The blocking parameter should be less than the number of columns
    b             = min(blocking,n)
    iteration     = 0
    # Update the maximal number of iterations according to b
    iteration_max = max(np.ceil(n/b),iteration_max) 
    norm_A_Fro = norm(A,'fro')
    
    while norm(A,'fro') > epsilon*norm_A_Fro and iteration < iteration_max:
        # Create R the random block based on Gaussian variables
        R      = np.random.randn(n,b)
        # Matrix-matrix product Y = AR 
        Y      = A@R
        # QR decomposition of Y
        QY, RY = qr(Y,mode='economic')
        # Compute the projection 
        BY     = QY.T@A
        # Concatenate the information related to Q and B
        if iteration == 0:
            Q  = QY
            B  = BY
        else:
            Q  = np.concatenate((Q, QY),axis=1)
            B  = np.concatenate((B, BY),axis=0)
        # Update the iteration count
        iteration += 1
        # Update of A
        A = A - QY@BY
        # Upper bound of epsilon-rank
        erank = Q.shape[1]
        
    return Q, Q.shape[1], norm(A,'fro')/norm_A_Fro, iteration

def column_ID(A, k):
    """
    Compute a column interpolative decomposition such that 
        A ≈ A[:, J] * Z

    via pivoted QR.  We want:
        A * P = Q * R
    with P a permutation matrix.  Then by taking the first k pivots
    as J, and defining T = R11^{-1} * R12, we can construct Z so that
        A ≈ A[:, J] * Z.

    Input
    -----
    A : ndarray of shape (m, n)
        Matrix to be analyzed.
    k : int
        Estimated rank of matrix A.

    Output
    ------
    J : ndarray of shape (k,)
        The set of chosen column indices.
    Z : ndarray of shape (k, n)
        The matrix satisfying A ≈ A[:, J] * Z.

    Reference: Section 10.3 and Figure 7 of Martinsson's tutorial
               [https://arxiv.org/pdf/1607.01649.pdf].
    """
    m, n = A.shape

    # Step 1: Compute the pivoted QR factorization.
    # The function returns Q, R, and an array of pivot indices (piv),
    # where the pivoting rearranges the columns of A.
    Q, R, piv = qr(A, pivoting=True, mode='economic')

    # Step 2: Split R into the top-left block (R11) and the top-right block (R12).
    R11 = R[:k, :k]        # The leading k-by-k block of R.
    R12 = R[:k, k:]        # The remaining block, with dimensions k x (n-k).

    # Step 3: Solve for T such that R11 * T = R12.
    # This computes the interpolation coefficients.
    T = np.linalg.solve(R11, R12)

    # Step 4: Select the first k pivot indices as the representative columns.
    J = piv[:k]

    # Step 5: Construct the interpolation matrix Z.
    Z = np.zeros((k, n))

    # For the pivot columns, set the corresponding positions to form an identity matrix.
    for i, col_idx in enumerate(J):
        Z[i, col_idx] = 1.0

    # For the remaining columns, fill in the interpolation coefficients from T.
    for i, col_idx in enumerate(piv[k:]):
        Z[:, col_idx] = T[:, i]

    return J, Z


def row_ID(A, k):
    """
    Compute a row interpolative decomposition such that 
        A ≈ X * A[I, :]

    via pivoted QR applied to A^T.  Concrètement, on cherche :
        A^T * P = Q * R
    où P est la matrice de permutation, puis on récupère les 'k' premières
    lignes pivotées de A (c'est-à-dire les 'k' premières colonnes pivotées de A^T).

    Input
    -----
    A : ndarray of shape (m, n)
        Matrice à approximer.
    k : int
        Rang estimé (ou nombre de lignes "importantes" à sélectionner).

    Output
    ------
    I : ndarray of shape (k,)
        Indices des lignes pivot (les plus significatives).
    X : ndarray of shape (m, k)
        Matrice permettant de reconstruire A par :
            A ≈ X * A[I, :]
    """
    m, n = A.shape

    # Step 1: Perform a pivoted QR factorization on the transpose of A.
    # Note: The QR factorization applied on A^T identifies the important rows of A.
    Q, R, piv = qr(A.T, pivoting=True, mode='economic')

    # Step 2: Extract the leading blocks from R.
    R11 = R[:k, :k]  # Leading k-by-k block.
    R12 = R[:k, k:]  # Block of size k x (m-k) corresponding to the remaining columns of A^T.

    # Step 3: Solve for T in R11 * T = R12 to get the interpolation coefficients.
    T = np.linalg.solve(R11, R12)

    # Step 4: Choose the first k pivot indices as the representative row indices.
    I = piv[:k]

    # Step 5: Build the matrix X for reconstructing A.
    X = np.zeros((m, k))

    # For rows selected in I, assign the identity.
    for local_idx, row_idx in enumerate(I):
        X[row_idx, local_idx] = 1.0

    # For the remaining rows, fill in the interpolation coefficients from T.
    for j, row_idx in enumerate(piv[k:]):
        X[row_idx, :] = T[:, j]

    return I, X

def double_sided_ID(A, k):
    """
    Compute a double-sided Interpolative Decomposition (ID) of the matrix A:
        A ≈ X * A[I, J] * Z
    
    where:
      - I is the set of row indices (length k)
      - J is the set of column indices (length k)
      - X is (m x k)
      - Z is (k x n)

    Steps:
      1) Column ID on A       -> A ≈ A[:, J] * Z
      2) Row ID on A[:, J]    -> A[:, J] ≈ X * A[I, J]

    By combining both:
      A ≈ X * A[I, J] * Z

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to be approximated.
    k : int
        Estimated rank (number of pivot rows/columns).

    Returns
    -------
    I : ndarray of shape (k,)
        Indices of pivot rows.
    J : ndarray of shape (k,)
        Indices of pivot columns.
    X : ndarray of shape (m, k)
        Row-interpolation matrix.
    Z : ndarray of shape (k, n)
        Column-interpolation matrix.

    Reference: Section 10.3 and Figure 7 of Martinsson's tutorial 
               [https://arxiv.org/pdf/1607.01649.pdf].
    """
    # Step 1: Perform column ID to approximate A as A[:, J]*Z.
    J, Z = column_ID(A, k)
    
    # Step 2: Apply row ID on the submatrix A[:, J] to get the important rows.
    sub_A = A[:, J]
    I, X = row_ID(sub_A, k)
    
    return I, J, X, Z

def extract_subblock(A, I, J):
    """
    Given a set of row and column indices, extract the submatrix 
    B = A[I, J]
    with I the set of row indices and J the set of column indices.
    
    Input: 
    A : array-like of shape (m, n)
        Matrix to be analyzed.
    I : array-like
        Set of row indices.
    J : array-like
        Set of column indices.
    
    Output:
    B : ndarray of shape (len(I), len(J))
        The extracted submatrix.
    """
    # Initialize an empty matrix B with dimensions based on I and J.
    B = np.zeros((len(I), len(J)))
    row_index = 0
    
    # Iterate over each row index in I.
    for i in I:
        col_index = 0
        # For each selected row, iterate over each column index in J.
        for j in J:
            # Assign the corresponding element from A to B.
            B[row_index, col_index] = A[i, j]
            col_index += 1  # Move to the next column in B.
        row_index += 1  # Move to the next row in B.
    
    return B

def cur(A, k):
    """
    Deterministic version of the CUR algorithm based on the 
    double-sided ID decomposition.

    We want:
        A ≈ C U R

    with:
      - C = A[:, J]  (m x k)
      - R = A[I, :]  (k x n)
      - U = some (k x k) matrix
    
    Steps:
      1) Obtain (I, J, X, Z) via double_sided_ID(A, k),
         so A ≈ X * A[I,J] * Z.

      2) Define:
         C = A[:, J],   R = A[I, :],   W = A[I, J].

      3) Solve for U (size k x k). The standard deterministic choice is
         U = pinv(W) = (A[I,J])^dagger.
         This ensures C U R = A[:, J] * pinv(A[I,J]) * A[I, :].

    Returns
    -------
    I : array of shape (k,)
        Row indices.
    J : array of shape (k,)
        Column indices.
    C : ndarray of shape (m, k)
        The matrix containing the selected columns of A.
    U : ndarray of shape (k, k)
        The computed linking matrix, using the pseudo-inverse of W.
    R : ndarray of shape (k, n)
        The matrix containing the selected rows of A.
    """
    # Step 1: Apply the double-sided ID to obtain the pivots and interpolation matrices.
    I, J, X, Z = double_sided_ID(A, k)

    # Step 2: Construct matrices C and R based on the selected column and row indices.
    C = A[:, J]   # Matrix made of the selected k columns.
    R = A[I, :]   # Matrix made of the selected k rows.

    # Step 3: Extract the submatrix W = A[I, J], which is the intersection of the selected rows and columns.
    W = extract_subblock(A, I, J)

    # Step 4: Compute U as the pseudo-inverse of W.
    # This pseudo-inverse provides the best least-squares solution if W is rank-deficient.
    U = np.linalg.pinv(W)

    return I, J, C, U, R

def randomized_row_ID(A, k, p=10, q=2):
    """
    Randomized version of the Row ID to decompose matrix A into
        A ≈ X * A[I, :]
    
    Input
    -----
    A : ndarray of shape (m, n)
        Matrix to be approximated.
    k : int
        Target rank (number of pivot rows to select).
    p : int, optional
        Oversampling parameter (default = 10).
    q : int, optional
        Number of power iterations (default = 2).
    
    Output
    ------
    I : ndarray of shape (k,)
        Indices of the selected pivot rows.
    X : ndarray of shape (m, k)
        Interpolation matrix such that A ≈ X * A[I, :].
    
    Reference: Section 10.4 and Figure 8 of Martinsson's tutorial 
               [https://arxiv.org/pdf/1607.01649.pdf].
    """
    m, n = A.shape

    # Step 1: Generate a random Gaussian matrix G of size (k+p) x m.
    # This matrix is used to project A's row space.
    G = np.random.normal(loc=0.0, scale=1.0, size=(k+p, m))

    # Step 2: Project A onto a lower-dimensional space by computing Y = A * G.
    # The resulting matrix Y has dimensions m x (k+p).
    Y = A @ G

    # Step 3: Perform power iterations to enhance the decay of singular values,
    # thereby improving the accuracy of the subspace approximation.
    for _ in range(q):
        Y = A @ (A.T @ Y)

    # Step 4: Apply a row interpolative decomposition on Y.
    # This step finds pivot row indices I and an interpolation matrix X such that Y ≈ X * Y[I, :].
    # Due to the properties of Y, the pivot rows also serve as a good approximation for A.
    I, X = row_ID(Y, k)

    return I, X

   
def randomized_cur(A, k, p=10, q=2):
    """
    Randomized version of the CUR algorithm.

    Input
    -----
    A : ndarray of shape (m, n)
        Matrix to be approximated.
    k : int
        Target rank (number of rows/columns to select).
    p : int, optional
        Oversampling parameter (default = 10).
    q : int, optional
        Number of power iterations (default = 2).

    Output
    ------
    I, J : ndarray of shape (k,)
           Indices of the selected pivot rows (I) and pivot columns (J).
    C : ndarray of shape (m, k)
        Matrix containing the selected pivot columns from A.
    U : ndarray of shape (k, k)
        The linking matrix (computed as the pseudoinverse of A[I, J]).
    R : ndarray of shape (k, n)
        Matrix containing the selected pivot rows from A.

    Reference: Section 11.2 and Figure 10 of Martinsson's tutorial 
               [https://arxiv.org/pdf/1607.01649.pdf].
    """
    m, n = A.shape

    # Step 1: Generate a random Gaussian matrix G of size (k+p) x m.
    # This helps approximate the column space of A.
    G = np.random.normal(loc=0.0, scale=1.0, size=(k+p, m))

    # Step 2: Compute the initial projection Y = G * A.
    # Y is an approximation of A's column space and has dimensions (k+p) x n.
    Y = G @ A

    # Step 3: Refine the subspace approximation by performing power iterations.
    # In each iteration, update Y = Y * (A^T A) to better capture the dominant columns.
    for _ in range(q):
        Y = Y @ (A.T @ A)

    # Step 4: Apply a column interpolative decomposition on Y to select k pivot columns.
    # This returns the indices of the important columns (J) and an interpolation matrix Z such that:
    # Y ≈ Y[:, J] * Z.
    # The selected columns J correspond directly to the important columns in A.
    J, Z = column_ID(Y, k)

    # Step 5: Extract the pivot columns from A using the indices J.
    subA = A[:, J]

    # Step 6: Apply a row interpolative decomposition on the submatrix to select k pivot rows.
    # This returns the indices I and an interpolation matrix X such that subA ≈ X * subA[I, :].
    I, X = row_ID(subA, k)

    # Step 7: Form the matrices C, R, and U for the CUR decomposition.
    C = subA          # C is A restricted to the selected pivot columns A[:, J].
    R = A[I, :]       # R is A restricted to the selected pivot rows A[I, :].
    W = extract_subblock(A, I, J)  # W is the intersection submatrix A[I, J].

    # Step 8: Compute U as the pseudoinverse of W.
    # This links the pivot columns and rows to reconstruct A as A ≈ C * U * R.
    U = np.linalg.pinv(W)

    return I, J, C, U, R


def low_rank_approximation(imgc, epsilon, blocking, randomization):
    """Low rank approximation of the RGB image using two algorithms:
        - CUR 
        - Randomized CUR
    """
    # Convert input to a numpy array for consistency.
    img = np.array(imgc)
    img_low_rank = np.array(imgc)
    eranks = []  # List to store the estimated epsilon-rank for each color channel.
    
    # Variables to accumulate CUR running time and Frobenius norm errors for each channel.
    total_CUR_time = 0.0
    frob_errors = []  # This list will hold the Frobenius errors for each color channel.

    #
    # Process the Red channel
    #
    t_start = time.time()
    # Determine an adaptive rank approximation for the red channel.
    Q, erank, norm_ratio, iteration = adaptive_rank_determination(img[:, :, 0], epsilon, blocking)
    print("Time spent determining the approximated epsilon-rank (R):", time.time() - t_start)
    print("Estimated upper bound of the epsilon-rank (R):", erank, "for relative threshold:", epsilon)
    print("Frobenius norm ratio (R):", norm_ratio, "reached in", iteration, "iterations")
    eranks.append(erank)
    
    t_start = time.time()
    # Compute the CUR decomposition on the red channel, using randomization if specified.
    if randomization:
        I, J, C, U, R = randomized_cur(img[:, :, 0], erank)
    else:
        I, J, C, U, R = cur(img[:, :, 0], erank)
    # Reconstruct the red channel using the CUR factors.
    JCUR_R = C @ U @ R
    t_cur_R = time.time() - t_start
    total_CUR_time += t_cur_R
    print("Time spent in the CUR algorithm for the R channel:", t_cur_R)
    
    # Compute the Frobenius norm error for the red channel.
    frob_error_R = norm(img[:, :, 0] - JCUR_R, 'fro')

    #
    # Process the Green channel
    #
    t_start = time.time()
    # Determine an adaptive rank approximation for the green channel.
    Q, erank, norm_ratio, iteration = adaptive_rank_determination(img[:, :, 1], epsilon, blocking)
    print("Time spent determining the approximated epsilon-rank (G):", time.time() - t_start)
    print("Estimated upper bound of the epsilon-rank (G):", erank, "for relative threshold:", epsilon)
    print("Frobenius norm ratio (G):", norm_ratio, "reached in", iteration, "iterations")
    eranks.append(erank)
    
    t_start = time.time()
    # Compute the CUR decomposition on the green channel.
    if randomization:
        I, J, C, U, R = randomized_cur(img[:, :, 1], erank)
    else:
        I, J, C, U, R = cur(img[:, :, 1], erank)
    # Reconstruct the green channel using the CUR factors.
    JCUR_G = C @ U @ R
    t_cur_G = time.time() - t_start
    total_CUR_time += t_cur_G
    print("Time spent in the CUR algorithm for the G channel:", t_cur_G)
    
    # Compute the Frobenius norm error for the green channel.
    frob_error_G = norm(img[:, :, 1] - JCUR_G, 'fro')

    #
    # Process the Blue channel
    #
    t_start = time.time()
    # Determine an adaptive rank approximation for the blue channel.
    Q, erank, norm_ratio, iteration = adaptive_rank_determination(img[:, :, 2], epsilon, blocking)
    print("Time spent determining the approximated epsilon-rank (B):", time.time() - t_start)
    print("Estimated upper bound of the epsilon-rank (B):", erank, "for relative threshold:", epsilon)
    print("Frobenius norm ratio (B):", norm_ratio, "reached in", iteration, "iterations")
    eranks.append(erank)
    
    t_start = time.time()
    # Compute the CUR decomposition on the blue channel.
    if randomization:
        I, J, C, U, R = randomized_cur(img[:, :, 2], erank)
    else:
        I, J, C, U, R = cur(img[:, :, 2], erank)
    # Reconstruct the blue channel using the CUR factors.
    JCUR_B = C @ U @ R
    t_cur_B = time.time() - t_start
    total_CUR_time += t_cur_B
    print("Time spent in the CUR algorithm for the B channel:", t_cur_B)
    
    # Compute the Frobenius norm error for the blue channel.
    frob_error_B = norm(img[:, :, 2] - JCUR_B, 'fro')

    #
    # Construct the low-rank approximation of the image by combining the channels.
    #
    img_low_rank[:, :, 0] = JCUR_R
    img_low_rank[:, :, 1] = JCUR_G
    img_low_rank[:, :, 2] = JCUR_B
    
    # Calculate the average Frobenius norm error over the three color channels.
    mean_frob_error = (frob_error_R + frob_error_G + frob_error_B) / 3.0

    # Return the low-rank approximated image, total CUR execution time, average error, and estimated ranks.
    return img_low_rank, total_CUR_time, mean_frob_error, eranks

def nmf_core(M, erank, iteration_max, epsilon):
    """
    Core function of the Non-negative Matrix Factorization (NMF) algorithm.

    This function factors the input matrix M into two non-negative matrices X and Y 
    (i.e. M ≈ X @ Y) using an iterative multiplicative update rule. The iterations stop 
    when the relative Frobenius norm of the error falls below epsilon or when 
    iteration_max is reached.

    Parameters
    ----------
    M : ndarray
        Input matrix to be factorized (typically representing a single image channel).
    erank : int
        Target rank for the approximation.
    iteration_max : int
        Maximum number of iterations allowed.
    epsilon : float
        Tolerance factor. The iterations terminate when the Frobenius norm 
        of the residual (M - X @ Y) falls below epsilon times the Frobenius norm of M.

    Returns
    -------
    X : ndarray
        Basis matrix of shape (m, erank).
    Y : ndarray
        Coefficient matrix of shape (erank, n).
    iteration : int
        Number of iterations performed.
    """
    m = np.shape(M)[0]  # Number of rows in M.
    n = np.shape(M)[1]  # Number of columns in M.

    # Initialize random generator with a fixed seed for reproducibility.
    rng = np.random.default_rng(seed=12345)
    # Print the random number generator object to verify initialization.
    print(rng)
    # Initialize the factor matrices X and Y with random entries.
    X = rng.random((m, erank))
    Y = rng.random((erank, n))
    
    # Compute the Frobenius norm of M for later convergence check.
    norm_A_Fro = norm(M, 'fro')
    iteration = 0

    # Iteratively update X and Y using multiplicative update rules.
    while norm(M - X @ Y, 'fro') > epsilon * norm_A_Fro and iteration < iteration_max:
        # Update Y:
        #   Y <- Y * (X^T * M) / (X^T * X * Y)
        # The element-wise multiplication and division ensure non-negativity.
        Y = np.multiply(Y, np.divide(np.dot(X.T, M), np.dot(np.dot(X.T, X), Y)))
        
        # Update X:
        #   X <- X * (M * Y^T) / (X * Y * Y^T)
        X = np.multiply(X, np.divide(np.dot(M, Y.T), np.dot(np.dot(X, Y), Y.T)))
        
        iteration += 1
        # Print intermediate information every 25 iterations for tracking the convergence.
        if iteration % 25 == 0:
            current_error = norm(M - X @ Y, 'fro') / norm_A_Fro
            print(iteration, current_error)

    return X, Y, iteration



def nmf(imgc, epsilon, blocking, iteration_max):
    """
    Perform Non-negative Matrix Factorization (NMF) on an RGB image.

    This function applies NMF to each color channel (Red, Green, Blue) independently.
    It first estimates an appropriate rank for each channel via an adaptive rank determination 
    procedure, and then factors each channel using the core NMF algorithm (nmf_core).

    Parameters
    ----------
    imgc : array-like
        Input RGB image.
    epsilon : float
        Relative threshold for the adaptive rank estimation and convergence of NMF.
    blocking : int
        Parameter used in the adaptive rank determination algorithm.
    iteration_max : int
        Maximum number of iterations allowed for NMF.

    Returns
    -------
    img_low_rank : ndarray
        The reconstructed low-rank approximated image.
    total_nmf_time : float
        Total time spent executing the nmf_core function across all channels.
    iter_counts : dict
        Dictionary containing the number of NMF iterations performed for each channel (keys: 'R', 'G', 'B').
    eranks : list
        List containing the estimated epsilon-rank for each channel (order: [R, G, B]).
    frob_error_avg : float
        The average Frobenius norm error between the original and the reconstructed image across the three channels.
    """
    # Convert input image to a numpy array.
    img = np.array(imgc)
    # Initialize the low-rank approximation with the original image.
    img_low_rank = np.array(imgc)
    total_nmf_time = 0.0
    eranks = []       # To store the estimated rank for each channel.
    iter_counts = {}  # To keep track of the iteration counts per channel.
    
    # Process the Red channel.
    t_start = time.time()
    # Estimate the epsilon-rank for the red channel using adaptive rank determination.
    Q, erank_R, norm_ratio_R, iteration_R = adaptive_rank_determination(img[:, :, 0], epsilon, blocking)
    print("Time spent in rank determination (R):", time.time() - t_start)
    print("Estimated epsilon-rank (R):", erank_R, "with Frobenius norm ratio:", norm_ratio_R, "in", iteration_R, "iterations")
    eranks.append(erank_R)
    
    t_channel = time.time()
    # Factorize the red channel using NMF.
    X_R, Y_R, iter_R_nmf = nmf_core(img[:, :, 0], erank_R, iteration_max, epsilon)
    t_R = time.time() - t_channel
    total_nmf_time += t_R
    iter_counts['R'] = iter_R_nmf

    # Process the Green channel.
    t_start = time.time()
    # Estimate the epsilon-rank for the green channel.
    Q, erank_G, norm_ratio_G, iteration_G = adaptive_rank_determination(img[:, :, 1], epsilon, blocking)
    print("Time spent in rank determination (G):", time.time() - t_start)
    print("Estimated epsilon-rank (G):", erank_G, "with Frobenius norm ratio:", norm_ratio_G, "in", iteration_G, "iterations")
    eranks.append(erank_G)
    
    t_channel = time.time()
    # Factorize the green channel using NMF.
    X_G, Y_G, iter_G_nmf = nmf_core(img[:, :, 1], erank_G, iteration_max, epsilon)
    t_G = time.time() - t_channel
    total_nmf_time += t_G
    iter_counts['G'] = iter_G_nmf

    # Process the Blue channel.
    t_start = time.time()
    # Estimate the epsilon-rank for the blue channel.
    Q, erank_B, norm_ratio_B, iteration_B = adaptive_rank_determination(img[:, :, 2], epsilon, blocking)
    print("Time spent in rank determination (B):", time.time() - t_start)
    print("Estimated epsilon-rank (B):", erank_B, "with Frobenius norm ratio:", norm_ratio_B, "in", iteration_B, "iterations")
    eranks.append(erank_B)
    
    t_channel = time.time()
    # Factorize the blue channel using NMF.
    X_B, Y_B, iter_B_nmf = nmf_core(img[:, :, 2], erank_B, iteration_max, epsilon)
    t_B = time.time() - t_channel
    total_nmf_time += t_B
    iter_counts['B'] = iter_B_nmf

    # Reconstruct each color channel from its factor matrices.
    rec_R = X_R @ Y_R
    rec_G = X_G @ Y_G
    rec_B = X_B @ Y_B
    
    # Combine the reconstructed channels to form the full color image.
    img_low_rank[:, :, 0] = rec_R
    img_low_rank[:, :, 1] = rec_G
    img_low_rank[:, :, 2] = rec_B

    # Compute the Frobenius norm error for each channel.
    frob_error_R = norm(img[:, :, 0] - rec_R, 'fro')
    frob_error_G = norm(img[:, :, 1] - rec_G, 'fro')
    frob_error_B = norm(img[:, :, 2] - rec_B, 'fro')
    # Calculate the average Frobenius error across all channels.
    frob_error_avg = (frob_error_R + frob_error_G + frob_error_B) / 3.0

    return img_low_rank, total_nmf_time, iter_counts, eranks, frob_error_avg



