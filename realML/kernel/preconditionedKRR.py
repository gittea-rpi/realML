import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import scipy.sparse.linalg
import numpy.linalg
from scipy.linalg.lapack import dpotrf
from numpy.fft import fft, ifft

def PCGfit(X, y, kernel, U, lparam, eps, maxIters): 
    n, d = X.shape
    _, t = y.shape

    Kreg = kernel(X, X) + lparam*np.eye(n) 
    coeffs = np.zeros((n, t))

    def approxInverse(v):
        return 1/lparam*(v - U.transpose().dot(U.dot(v)))
    rfmPrecond = scipy.sparse.linalg.LinearOperator((n, n), 
                                                    matvec=approxInverse)

    itercount = [0]
    def count(v):
        itercount[0] += 1

    # TODO: only use on PCG at once for all targets
    for tdim in range(t):
        x, info = scipy.sparse.linalg.cg(Kreg, y[:, tdim], tol=eps, 
                                         maxiter=maxIters, 
                                         M=rfmPrecond, callback = count)
        #print "iters for precond:", itercount[0]
        #itercount[0] = 0
        #x, info = scipy.sparse.linalg.cg(Kreg, y[:, tdim], tol=self.eps, maxiter=self.maxIters, callback = count)
        #print "iters for nonprecond:", itercount[0]
        coeffs[:, tdim] = x

    return coeffs

def GaussianKernel(Xrows, Xcols, sigma):
    """
    Computes the Gaussian kernel matrix K_{ij} = exp(-||xrows_i - xcols_j||_2^2/(2sigma^2))

    Inputs:
        Xrows: array [n_samples_row, n_features]
        Xcols: array [n_samples_col, n_features]

    Output:
        K: array of kernel evaluations [n_samples_row, n_samples_col]
    """
    return rbf_kernel(Xrows, Xcols, gamma=1/(2*sigma**2))


def PolynomialKernel(Xrows, Xcols, sf, offset, degree):
    """
    Computes the polynomial kernel matrix K_{ij} = (sf<x_i, x_j> + offset)^degree

    Inputs:
        Xrows: array [n_samples_row, n_features]
        Xcols: array [n_samples_col, n_features]

    Output:
        K: array of kernel evaluations [n_samples_row, n_samples_col]
    """
    return polynomial_kernel(Xrows, Xcols, degree=degree, 
                             gamma=sf, coef0=offset)

def findStableRank(X):
    # TODO: implement one or both estimators from Woodruff's work
    return 2*max(150, X.shape[1])

def generateGaussianPreconditioner(X, sigma, lparam):
    s = findStableRank(X)
    d = X.shape[1]
    Z = np.sqrt(2.0/s)*np.cos(X.dot(np.random.randn(d, s))/sigma
                                    + 2*np.pi*np.random.rand(1,s))
    L = scipy.linalg.cholesky(Z.transpose().dot(Z) + lparam*np.identity(s))
    return numpy.linalg.solve(L, Z.transpose())

def generatePolynomialPreconditioner(X, sf, offset, degree, lparam):
    """
    computes the preconditioner for a polynomial kernel using TensorSketch 
    """
    # TODO: use CraftMAps

    s = findStableRank(X)
    (n,d) = X.shape

    augXt = np.vstack((np.sqrt(sf)*X.transpose(),
                       np.sqrt(offset)*np.ones((1, n))))
    Z = np.ones((s, n))
    for iter in range(degree):
        C = np.zeros((s, d+1))
        rows = numpy.random.choice(s, size=d+1, replace=True)
        signs = numpy.random.choice([-1.0, 1.0], size=d+1, replace=True)
        for col in range(d+1):
            C[rows[col], col] = signs[col]
        Z = Z * fft(C.dot(augXt), axis=0)
    Z = ifft(Z, axis=0).real.transpose()
    #K = self.kernel(X, X)
    #print(numpy.linalg.norm(K), numpy.linalg.norm(Z.dot(Z.transpose())))

    precondMat = Z.transpose().dot(Z) + lparam*np.identity(s)
    L = dpotrf(precondMat)[0].transpose()
    return numpy.linalg.solve(L, Z.transpose())

