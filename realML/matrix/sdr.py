"""
Sufficient Dimensionality Reduction: 
    given an m-by-n matrix G of coocurrence statistics for discrete random variables X (m states) and Y (n states)
    and a desired embedding size k, returns U and V, m-by-k and n-by-k matrices of embeddings for the states of 
    X and Y minimizing D_KL(G/Zg || exp(UV^T)/Z) where Z and Zg are constants that normalize both matrices
    to be probability distributiions

This implementation uses Adagrad to solve the optimiztion problem, and uses 
random features to make the gradient computation much more scalable than the algorithm given in the SDR paper.
Further, it initializes with a factorization of the pointwise mutual information matrix.

see:
Globerson and Tishby. "Sufficient dimensionality reduction." JMLR, 2003
Gittens, Achlioptas, and Mahoney. "Skip-Gram - Zipf + Uniform = Vector Additivity". ACL, 2017
Le et al., "Fastfood --- Approximating Kernel Expansions in Loglinear Time", ICML 2013


"""

# Author: Alex Gittens <gittea@rpi.edu>
# License: TBD

import abc
from .featurization import FeaturizationPrimitiveBase
from numpy.random import randn, random
from numpy import pi, exp, cos, sqrt, expand_dims, sum, ones_like
from numpy.linalg import norm

__all__ = ['SDR']

class SDR(FeaturizationPrimitiveBase):

    def __init__(self, dim=300, numrandfeats=5000, maxIters=50, tol=.01, stepsize=.1, eps=.001):
        """
        inputs:
            dim: the desired dimensionality of the embeddings (positive integer)
            numrandfeats: size of the random feature map used in estimating the gradient (positive integer)
            maxIters: maximum number of gradient steps to take (positive integer)
            tol: stop when relative change (Frobenius norm) of embeddings is smaller than tol
            stepsize: Adagrad stepsize
            eps: Adagrad protection against division by zero, small constant
        """
        self.dim = dim
        self.numrandfeats = numrandfeats
        self.maxIters = maxIters
        self.tol = tol
        self.stepsize = stepsize
        self.eps = eps

        return 

    def fit(self, intype="matrix", data=None, labels=None):
        """
        inputs:
            data: a 2d (sparse or dense) numpy array of co-occurence statistics (non-negative)

        internally computes and sets U and V, the embeddings of the row and column entities, respectively
        """

        # TODO: initialize with the pmi matrix

        m, n = data.shape
        self.U = randn(m, self.dim)/sqrt(self.dim)
        self.V = randn(n, self.dim)/sqrt(self.dim)
        gUhist = self.eps**2 * ones_like(self.U)
        gVhist = self.eps**2 * ones_like(self.V)

        for iter in range(self.maxIters):
            Unorms = expand_dims(norm(self.U, axis=1)**2, axis=1)
            Vnorms = expand_dims(norm(self.V, axis=1)**2, axis=1)
            W = randn(self.dim, self.numrandfeats)
            phases =2*pi*random([1, self.numrandfeats])
            ZU = exp(1.0/2*Unorms) * sqrt(2.0/self.numrandfeats)*cos(self.U.dot(W) + phases)
            ZV = exp(1.0/2*Vnorms) * sqrt(2.0/self.numrandfeats)*cos(self.V.dot(W) + phases)

            v1 = sum(ZV, axis=0)
            v2 = sum(ZU, axis=0)
            normalizer = 1.0/(v1.dot(v2)) # compute 1^T ZU ZV^T 1
            gU = -1 * data.dot(self.V) + normalizer * ZU.dot(ZV.transpose().dot(self.V))
            gV = -1 * data.transpose().dot(self.U) + normalizer * ZV.dot(ZU.transpose().dot(self.U))

            gUhist += gU**2
            gVhist += gV**2
            dU = gU / sqrt(gUhist)
            dV = gV / sqrt(gVhist)
            self.U = self.U - self.stepsize*dU
            self.V = self.V - self.stepsize*dV

    def predict(self, outtype="array2+N", data=None):
        """
        inputs:
            outtype: "array2+N" where N=2
            data: a 2d (sparse or dense) numpy array of co-occurrence statistics (non-negative)

        outputs:
            [2, U, V], where
            U: each row of U is an embedding for the entity corresponding to that row of the co-occurence matrix
            U: each row of V is an embedding for the entity corresponding to that column of the co-occurence matrix
        """
        if data is not None:
            self.fit("matrix", data)
        return [2, self.U, self.V]
