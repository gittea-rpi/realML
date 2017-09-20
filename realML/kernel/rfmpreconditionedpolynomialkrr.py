from typing import *
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from sklearn.metrics.pairwise import polynomial_kernel
import scipy.sparse.linalg
from numpy.fft import fft, ifft
import numpy as np
import numpy.random
import numpy.linalg 

Input=np.ndarray
Output=np.ndarray
Params = NamedTuple('Params', [ ])


__all__ = ('RFMPreconditionedPolynomialKRR')

class RFMPreconditionedPolynomialKRR(SupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    Performs polynomial kernel regression using a random feature map to precondition the
    problem for faster convergence: takes
    - data X
    - targets y 
    - regularization parameter lambda
    - scaling parameter sf
    - offset parameter offset
    - degree parameter degree,
    forms the kernel 
        K_{ij} = (sf<x,y>+offset)^degree
    and solves 
        alphahat = argmin ||K alpha - y||_F^2 + lambda ||alpha||_F^2 
    predictions are then formed by 
        ypred = K(trainingData, x) alphahat
    """

    def __init__(self, *, lparam: float = None, degree: int = 3, offset: float = 1.0, sf: float = None, eps: float = 1e-05) -> None:
        """
        Initializes the preconditioned gaussian kernel ridge regression primitive.

        Inputs:
            lparam: numeric, the regularization parameter. if None, chooses lambda to be 1/sqrt(numsamples)
            degree: integer. degree of the polynomial. if None, choooses to be 3
            offset: numeric. offset parameter for the kernel. if None, chooses offset to be 1
            sf: numeric. scaling factor for the kernel. if None, chooses sf to be 1/numfeatures
            eps: numeric. termination accuracy
        """
        self.lparam = lparam 
        self.degree = degree
        self.offset = offset + 0.0
        self.sf = sf
        self.eps = eps + 0.0
        self.maxIters = None
    
    def set_training_data(self, *, inputs: Sequence[Input], outputs: Sequence[Output]) -> None:
        """
        Sets the training data:
            Input: array, shape = [n_samples, n_features]
            Output: array, shape = [n_samples, n_targets]
        Only uses one input and output
        """
        self.Xtrain = inputs[0]
        self.ytrain = outputs[0]
        self.fitted = False

        maxPCGsize = 20000

        if len(self.ytrain.shape) == 1:
            self.ytrain = np.expand_dims(self.ytrain, axis=1) 

        if self.Xtrain.shape[0] > maxPCGsize:
            print("need to implement Gauss-Siedel for large datasets; currently training with a smaller subset")
            choices = np.random.choice(self.Xtrain.shape[0], size=maxPCGsize, replace=False)
            self.Xtrain = self.Xtrain[choices, :] 
            self.ytrain = self.ytrain[choices, :]

        self.n, self.d = self.Xtrain.shape
        if (self.lparam is None) or (self.lparam == 0):
            self.lparam = 1/sqrt(self.n)
        if (self.sf is None):
            self.sf = 1.0/self.d
        if (self.offset is None):
            self.offset = 1
        if (self.degree is None):
            self.degree = 3

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        Learns the kernel regression coefficients alpha given training pairs (X,y)
        """
        if (iterations is not None):
            self.maxIters = iterations
        self.generatePreconditioner(self.Xtrain)
        self.PCGfit(self.Xtrain, self.ytrain)

    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
        """
        Predict the value for each sample in X

        Inputs:
            X: array or sparse matrix of shape [n_samples, n_features]
        Outputs:
            y: array of shape [n_samples, n_targets]
        """
        return [self.kernel(X, self.Xtrain).dot(self.alpha) for X in inputs]

    def PCGfit(self, X, y):
        n, d = X.shape
        _, t = y.shape

        Kreg = self.kernel(X, X) + self.lparam*np.eye(n)
        self.alpha = np.zeros((n, t))

        def approxInverse(v):
            return 1/self.lparam*(v - self.U.transpose().dot(self.U.dot(v)))
        rfmPrecond = scipy.sparse.linalg.LinearOperator((self.n,self.n), matvec=approxInverse)

        itercount = [0]
        def count(v):
            itercount[0] += 1

        for tdim in range(t):
            x, info = scipy.sparse.linalg.cg(Kreg, y[:, tdim], tol=self.eps, maxiter=self.maxIters, M=rfmPrecond, callback = count)
            #print "iters for precond:", itercount[0]
            #itercount[0] = 0
            #x, info = scipy.sparse.linalg.cg(Kreg, y[:, tdim], tol=self.eps, maxiter=self.maxIters, callback = count)
            #print "iters for nonprecond:", itercount[0]
            self.alpha[:, tdim] = x

    def kernel(self, Xrows, Xcols):
        """
        Computes the polynomial kernel matrix K_{ij} = (sf<x_i, x_j> + offset)^degree

        Inputs:
            Xrows: array [n_samples_row, n_features]
            Xcols: array [n_samples_col, n_features]

        Output:
            K: array of kernel evaluations [n_samples_row, n_samples_col]
        """
        return polynomial_kernel(Xrows, Xcols, degree=self.degree, gamma=self.sf, coef0=self.offset)

    def findStableRank(self, X):
        # TODO implement, and return the stable rank or a reasonable smaller number
        return 2*max(150, self.d)

    def generatePreconditioner(self, X):
        """
        computes the preconditioner for a polynomial kernel using TensorSketch 
        """
        # TODO: use CraftMAps

        self.s = self.findStableRank(X)
        augXt = np.vstack((np.sqrt(self.sf)*X.transpose(), np.sqrt(self.offset)*np.ones((1, self.n))))
        Z = np.ones((self.s, self.n))
        for iter in range(self.degree):
            C = np.zeros((self.s, self.d+1))
            rows = numpy.random.choice(self.s, size=self.d+1, replace=True)
            signs = numpy.random.choice([-1.0, 1.0], size=self.d+1, replace=True)
            for col in range(self.d+1):
                C[rows[col], col] = signs[col]
            Z = np.multiply(Z, fft(C.dot(augXt), axis=0, norm="ortho"))
        Z = np.real(np.sqrt(1.0/self.s)*ifft(Z, axis=0, norm="ortho").transpose())
        L = scipy.linalg.cholesky(Z.transpose().dot(Z) + self.lparam*np.identity(self.s))
        self.U = numpy.linalg.solve(L, Z.transpose())

        #K = self.kernel(X, X)
        #print numpy.linalg.norm(K), numpy.linalg.norm(Z.dot(Z.transpose()))

    def set_params(self, *, params: Params) -> None:
        pass

    def get_params(self) -> Params:
        return NamedTuple('Params', [])
