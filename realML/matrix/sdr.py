"""
Sufficient Dimensionality Reduction: 
    given an m-by-n matrix G of coocurrence statistics for discrete random variables X (m states) and Y (n states)
    and a desired embedding size k, returns U and V, m-by-k and n-by-k matrices of embeddings for the states of 
    X and Y minimizing D_KL(G/Zg || exp(UV^T)/Z) where Z and Zg are constants that normalize both matrices
    to be probability distributiions

This implementation uses Adagrad to solve the optimiztion problem, and uses 
random features to make the gradient computation much more scalable than the algorithm given in the SDR paper.

see:
Globerson and Tishby. "Sufficient dimensionality reduction." JMLR, 2003
Gittens, Achlioptas, and Mahoney. "Skip-Gram - Zipf + Uniform = Vector Additivity". ACL, 2017
Le et al., "Fastfood --- Approximating Kernel Expansions in Loglinear Time", ICML 2013

"""

# Author: Alex Gittens <gittea@rpi.edu>
# License: TBD

from typing import *
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase 
from numpy.random import randn, random
from numpy import pi, exp, cos, sqrt, expand_dims, sum, ones, ones_like, log, hstack, ndarray
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import sys

Input=ndarray
Output=ndarray
Params = NamedTuple('Params', [])

__all__ = ['SDR']

class SDR(UnsupervisedLearnerPrimitiveBase[Input, Output, Params]):
    """
    Sufficient Dimensionality Reduction: 
        given an m-by-n sparse matrix G of coocurrence statistics for discrete random variables X (m states) and Y (n states)
        and a desired embedding size k, returns U and V, m-by-k and n-by-k matrices of embeddings for the states of 
        X and Y minimizing D_KL(G/Zg || exp(UV^T)/Z) where Z and Zg are constants that normalize both matrices
        to be probability distributiions

    This implementation uses Adagrad to solve the optimization problem, and uses 
    random features to make the gradient computation much more scalable than the algorithm given in the SDR paper.

    Read docstring for __init__() to see hyperparameters, then use fit() and predict() (see their docstrings)
    """

    def __init__(self, *, dim: int = 300, numrandfeats: int = 1000, tol: float = .01, stepsize: float = 0.1, maxIters: int = 100, eps: float = 0.001):
        """
        inputs:
            dim: the desired dimensionality of the embeddings (positive integer)
            numrandfeats: size of the random feature map used in estimating the gradient (positive integer)
            tol: stop when relative change (Frobenius norm) of embeddings is smaller than tol
            stepsize: Adagrad stepsize
            maxIters: maximum number of iterations
            eps: Adagrad protection against division by zero, small constant
        """
        self.dim = dim
        self.numrandfeats = numrandfeats
        self.maxIters = maxIters
        self.tol = tol
        self.stepsize = stepsize
        self.eps = eps
        self.fitted = False

        return 

    def set_training_data(self, *, inputs: Sequence[Input], outputs: None = None) -> None:
        """
        Inputs:
            X : array, shape = [n_rows, n_cols]
        only takes one input
        """
        self.G = inputs[0]
        self.fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None: 
        """
        internally computes and sets U and V, the embeddings of the row and column entities, respectively
        """
        if self.fitted:
            return
        m, n = self.G.shape

        # initialize with the PPMI: force sparsity if the data was not originally sparse
        sparseData = coo_matrix(self.G)
        i = sparseData.row
        j = sparseData.col
        vals = sparseData.data

        numPairs = sum(vals) + 0.0
        numRowEntities = sum(self.G, axis=1) + 0.0
        numColEntities = sum(self.G, axis=0) + 0.0

        print("Constructing PPMI")
        ppmiEntries=vals
        for idx in range(len(vals)):
            if idx % 1000 == 0:
                sys.stdout.write(str(idx/1000.0)+'.')
            ppmiEntries[idx] = max(log(numPairs*vals[idx]/(numRowEntities[i[idx]] * numColEntities[j[idx]])), 0)
        sparsePPMI = coo_matrix((ppmiEntries, (i, j)))
        self.U, _, vt = svds(sparsePPMI, k=self.dim-1)
        self.V = vt.transpose()

        gUhist = self.eps**2 * ones_like(self.U)
        gVhist = self.eps**2 * ones_like(self.V)

        print("Refining")
        self.maxIters = 1 # TODO: fix Adagrad-- this is not working at all
        for iter in range(self.maxIters):
            sys.stdout.write(str(iter)+'.')
            sys.stdout.flush()
            Unorms = expand_dims(norm(self.U, axis=1)**2, axis=1)
            Vnorms = expand_dims(norm(self.V, axis=1)**2, axis=1)
            W = randn(self.dim-1, self.numrandfeats)
            phases =2*pi*random([1, self.numrandfeats])
            ZU = exp(1.0/2*Unorms) * sqrt(2.0/self.numrandfeats)*cos(self.U.dot(W) + phases)
            ZV = exp(1.0/2*Vnorms) * sqrt(2.0/self.numrandfeats)*cos(self.V.dot(W) + phases)

            v1 = sum(ZV, axis=0)
            v2 = sum(ZU, axis=0)
            normalizer = 1.0/(v1.dot(v2)) # compute 1^T ZU ZV^T 1
            gU = -1 * self.G.dot(self.V) + normalizer * ZU.dot(ZV.transpose().dot(self.V))
            gV = -1 * self.G.transpose().dot(self.U) + normalizer * ZV.dot(ZU.transpose().dot(self.U))

            gUhist += gU**2
            gVhist += gV**2
            dU = gU / sqrt(gUhist)
            dV = gV / sqrt(gVhist)
            if (norm(dU) < self.tol and norm(dV) < self.tol):
                print("reached convergence tolerance, terminating")
                break
            else:
                self.U = self.U - self.stepsize*dU
                self.V = self.V - self.stepsize*dV
        print(" ")

        # augment embeddings with the entity frequencies
        #rowNormalizers = sum(exp(self.U.dot(self.V.transpose())), axis=1)
        #rowProbs = numRowEntities/numPairs
        #self.U = hstack((self.U, expand_dims(log(rowProbs/rowNormalizers), axis=1)))
        #self.V = hstack((self.V, ones((n,1))))
        #self.fitted = True

    def produce(self, *, inputs: None = None, timeout: float = None, iterations : int = None) -> Sequence[Output]:
        """
        takes no inputs, returns the U,V embeddings for the rows/columns respectively 
        """
        return [self.U, self.V]

    def set_params(self, *, params: Params) -> None:
        pass

    def get_params(self) -> Params:
        return NamedTuple('Params', [])
