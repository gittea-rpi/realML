import typing
from typing import Any, List, Dict, Union, Optional, Sequence
from collections import OrderedDict
from numpy import ndarray
import os
import warnings

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import scipy.sparse.linalg
import numpy.linalg

from .preconditionedKRR import *

from d3m import utils
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive as NDArrayToDataFrame
from common_primitives.ndarray_to_dataframe import Hyperparams as NDArrayToDataFrameHyperparams

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    exemplars: Optional[ndarray] 
    coeffs : Optional[ndarray]

class Hyperparams(hyperparams.Hyperparams):
    # search over these hyperparameters to tune performance
    lparam = hyperparams.LogUniform(default=.01, lower=.0001, upper=1000, 
                                    description="l2 regularization to use for the kernel regression", 
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    sigma = hyperparams.LogUniform(default=.01, lower=.0001, upper=1000, 
                                   description="bandwidth (sigma) parameter for the kernel regression", 
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # control parameters determined once during pipeline building then fixed
    eps = hyperparams.LogUniform(default=1e-4, lower=1e-14, upper=1e-2, 
                                 description="relative error stopping tolerance for PCG solver", 
                                 semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    maxIters = hyperparams.UniformInt(default=200, lower=50, upper=500, 
                                      description="maximum iterations of PCG", 
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

# TO DO: normalize the objective so lparam and sigma don't need a data-dependent range!
class RFMPreconditionedGaussianKRR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Performs gaussian kernel regression using a random feature map to precondition the
    problem for faster convergence:
    forms the kernel 
        K_{ij} = exp(-||x_i - x_j||^2/(2sigma^2)) 
    and solves 
        alphahat = argmin ||K alpha - y||_F^2 + lambda ||alpha||_F^2 
    predictions are then formed by 
        ypred = K(trainingData, x) alphahat
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': '90d9eefc-2db3-4738-a0e7-72eedab2d93a',
        'version': __version__,
        'name': 'RFM Preconditioned Gaussian Kernel Ridge Regression',
        'description': 'Gaussian regression using random fourier features as a preconditioner for faster solves',
        'python_path': 'd3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR',
        'primitive_family': metadata_base.PrimitiveFamily.REGRESSION,
        'algorithm_types' : [
            'KERNEL_METHOD',
            'MULTIVARIATE_REGRESSION'
        ],
        'keywords' : ['kernel learning', 'kernel ridge regression', 'preconditioned CG', 'Gaussian', 'RBF', 'regression'],
        'source' : {
            'name': __author__,
            'contact': 'mailto:gittea@rpi.edu',
            'uris' : [
                'https://github.com/ICSI-RealML/realML.git',
            ],
        },
        'installation': [
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/ICSI-RealML/realML.git@{git_commit}#egg=realML'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ],
        'location_uris': [ # NEED TO REF SPECIFIC COMMIT
            'https://github.com/ICSI-RealML/realML/blob/master/realML/kernel/RFMPreconditionedGaussianKRRSolver.py',
            ],
        'preconditions': [
            'NO_MISSING_VALUES',
            'NO_CATEGORICAL_VALUES'
        ],
    })

    def __init__(self, *, 
                 hyperparams : Hyperparams,
                 random_seed: int = 0,
                 docker_containers : Dict[str, DockerContainer] = None) -> None:
        """
        Initializes the preconditioned gaussian kernel ridge regression primitive.
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._seed = random_seed
        self._Xtrain = None
        self._ytrain = None
        self._fitted = False
        np.random.seed(random_seed)
    
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets the training data:
            Input: array, shape = [n_samples, n_features]
            Output: array, shape = [n_samples, n_targets]
        Only uses one input and output
        """
        self._Xtrain = inputs
        self._ytrain = outputs
        self._ymetadata = outputs.metadata

        maxPCGsize = 20000 # TODO: make a control hyperparameter for when to switch to using Gauss-Siedel

        if len(self._ytrain.shape) == 1:
            self._ytrain = np.expand_dims(self._ytrain, axis=1) 

        if self._Xtrain.shape[0] > maxPCGsize:
            print("need to implement Gauss-Siedel for large datasets; currently training with a smaller subset")
            choices = np.random.choice(self._Xtrain.shape[0], size=maxPCGsize, replace=False)
            self._Xtrain = self._Xtrain[choices, :] 
            self._ytrain = self._ytrain[choices, :]

        self._n, self._d = self._Xtrain.shape
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Learns the kernel regression coefficients alpha given training pairs (X,y)
        """
        if self._fitted:
            return CallResult(None)

        if self._Xtrain is None or self._ytrain is None:
            raise ValueError("Missing training data.")

        self._U = generateGaussianPreconditioner(self._Xtrain, self.hyperparams['sigma'],
                                                 self.hyperparams['lparam'])
        def mykernel(X, Y):
            return GaussianKernel(X, Y, self.hyperparams['sigma'])
        self._coeffs = PCGfit(self._Xtrain, self._ytrain, mykernel, self._U, self.hyperparams['lparam'],
                              self.hyperparams['eps'], self.hyperparams['maxIters'])
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Predict the value for each sample in X

        Inputs:
            X: array of shape [n_samples, n_features]
        Outputs:
            y: array of shape [n_samples, n_targets]
        """
        result = d3m_ndarray(GaussianKernel(inputs, self._Xtrain, self.hyperparams['sigma']).dot(self._coeffs).flatten())
        result.metadata = self._ymetadata.set_for_value(result)
        return CallResult(result)

    def set_params(self, *, params: Params) -> None:
        self._Xtrain = params['exemplars']
        self._coeffs = params['coeffs']

    def get_params(self) -> Params:
        return Params(exemplars=self._Xtrain, coeffs=self._coeffs)
