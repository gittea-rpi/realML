from typing import *
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from sklearn.metrics.pairwise import rbf_kernel
import scipy.sparse.linalg
import numpy as np
import numpy.random
import numpy.linalg 

Input=np.ndarray
Output=np.ndarray
Params= NamedTuple('Params', [])

__all__ = ('RFMPreconditionedGaussianKRR')

class RFMPreconditionedGaussianKRR(SupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    Performs gaussian kernel regression using a random feature map to precondition the
    problem for faster convergence: takes
    - data X
    - targets y 
    - regularization parameter lambda
    - bandwidth parameter sigma,
    forms the kernel 
        K_{ij} = exp(-||x_i - x_j||^2/(2sigma^2)) 
    and solves 
        alphahat = argmin ||K alpha - y||_F^2 + lambda ||alpha||_F^2 
    predictions are then formed by 
        ypred = K(trainingData, x) alphahat
    """

    def __init__(self, *, lparam: float = 1, sigma: float = None , eps: float = 1e-05) -> None:
        """
        Initializes the preconditioned gaussian kernel ridge regression primitive.

        Inputs:
            lparam: numeric, the regularization parameter. if None, chooses lambda to be 1/sqrt(numsamples)
            sigma: numeric. bandwidth sigma for the kernel. if None or 0, chooses sigma to be sqrt(numfeatures/2) to match sklearn
            eps: numeric. termination accuracy
        """
        self.lparam = lparam + 0.0
        self.sigma = sigma 
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
        if (self.sigma is None) or (self.sigma == 0):
            self.sigma = sqrt(self.d/2)
        else:
            self.sigma = self.sigma + 0.0


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
        Computes the Gaussian kernel matrix K_{ij} = exp(-||xrows_i - xcols_j||_2^2/(2sigma^2))

        Inputs:
            Xrows: array [n_samples_row, n_features]
            Xcols: array [n_samples_col, n_features]

        Output:
            K: array of kernel evaluations [n_samples_row, n_samples_col]
        """
        return rbf_kernel(Xrows, Xcols, gamma=1/(2*self.sigma**2))

    def findStableRank(self, X):
        # TODO: implement one or both estimators from Woodruff's work
        return 2*max(150, self.d)

    def generatePreconditioner(self, X):
        self.s = self.findStableRank(X)
        Z = np.sqrt(2.0/self.s)*np.cos(X.dot(np.random.randn(self.d, self.s))/self.sigma + 2*np.pi*np.random.rand(1,self.s))
        L = scipy.linalg.cholesky(Z.transpose().dot(Z) + self.lparam*np.identity(self.s))
        self.U = numpy.linalg.solve(L, Z.transpose())

    def set_params(self, *, params: Params) -> None:
        pass

    def get_params(self) -> Params:
        return NamedTuple('Params', [])
