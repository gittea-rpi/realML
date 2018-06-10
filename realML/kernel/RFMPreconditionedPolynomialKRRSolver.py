import typing
from typing import Any, List, Dict, Union, Optional, Sequence
from collections import OrderedDict
from numpy import ndarray
import os

import numpy as np
import scipy.sparse.linalg
import numpy.linalg

from .preconditionedKRR import *

from d3m import utils
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    exemplars: Optional[ndarray] 
    coeffs : Optional[ndarray]

class Hyperparams(hyperparams.Hyperparams):
    # search over these hyperparameters to tune performance
    lparam = hyperparams.LogUniform(default=.01, lower=.0001, upper=1000, 
                                    description="l2 regularization to use on the regression",
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    degree = hyperparams.UniformInt(default=3, lower=2, upper=9, 
                                    description="degree of the polynomial to fit",
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    offset = hyperparams.LogUniform(default=.1, lower=.001, upper=2, 
                                    description="value of constant feature to use in the regression",
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    sf = hyperparams.LogUniform(default=.01, lower=.00001, upper=2, 
                                description="scale factor to use in the regression",
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # control parameters determined once during pipeline building then fixed
    eps = hyperparams.LogUniform(default=1e-3, lower=1e-14, upper=1e-2, 
                                 description="relative error stopping tolerance for PCG solver", 
                                 semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    maxIters = hyperparams.UniformInt(default=200, lower=50, upper=500, 
                                      description="maximum iterations of PCG", 
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

# TODO: Normalize problem being solved so lparam etc easy to cross validate over fixed range regardless of amount of training data
class RFMPreconditionedPolynomialKRR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Performs polynomial kernel regression using a TensorSketch polynomial random feature map to precondition the
    problem for faster convergence: 
    forms the kernel 
        K_{ij} = (sf<x,y>+offset)^degree
    and solves 
        alphahat = argmin ||K alpha - y||_F^2 + lambda ||alpha||_F^2 
    predictions are then formed by 
        ypred = K(trainingData, x) alphahat

    Warning: the data should be normalized (e.g. have every row of X be very low l2 norm), or numerical issues will arise when the degree is greater than 2
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'c7a35a32-444c-4530-aeb4-e7a95cbe2cbf',
        'version': __version__,
        'name': 'RFM Preconditioned Polynomial Kernel Ridge Regression',
        'description': 'Polynomial regression using random polynomial features as a preconditioner for faster solves',
        'python_path': 'd3m.primitives.realML.RFMPreconditionedPolynomialKRR',
        'primitive_family': 'REGRESSION',
        'algorithm_types' : [
            'KERNEL_METHOD',
            'MULTIVARIATE_REGRESSION'
        ],
        'keywords' : ['kernel learning', 'kernel ridge regression', 'preconditioned CG', 'polynomial', 'regression'],
        'source' : {
            'name': __author__,
            'contact': 'mailto:gittea@rpi.edu',
            'uris' : [
                "https://github.com/ICSI-RealML/realML.git",
            ],
        },
        'installation': [
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/ICSI-RealML/realML.git@{git_commit}#egg=realML'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ],
        'location_uris': [ # NEED TO REF SPECIFIC COMMIT
            'https://github.com/ICSI-RealML/realML/blob/master/realML/kernel/RFMPreconditionedPolynomialKRRSolver.py',
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
        Initializes the preconditioned polynomial kernel ridge regression primitive.
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._seed = random_seed
        np.random.seed(random_seed)

        self._Xtrain = None
        self._ytrain = None
        self._fitted = False
    
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets the training data:
            Input: array, shape = [n_samples, n_features]
            Output: array, shape = [n_samples, n_targets]
        Only uses one input and output
        """
        self._Xtrain = inputs
        self._ytrain = outputs
        self._fitted = False

        maxPCGsize = 20000 # TODO: make a control hyperparameter for when to switch to GS

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

        self._U = generatePolynomialPreconditioner(self._Xtrain, self.hyperparams['sf'],
                                                   self.hyperparams['offset'], self.hyperparams['degree'],
                                                   self.hyperparams['lparam'])
        def mykernel(X, Y):
            return PolynomialKernel(X, Y, self.hyperparams['sf'], self.hyperparams['offset'], 
                                    self.hyperparams['degree'])
        self._coeffs = PCGfit(self._Xtrain, self._ytrain, mykernel, self._U, 
                    self.hyperparams['lparam'], self.hyperparams['eps'],
                    self.hyperparams['maxIters'])
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
        return CallResult(PolynomialKernel(inputs, self._Xtrain, self.hyperparams['sf'],
                self.hyperparams['offset'], self.hyperparams['degree']).dot(self._coeffs).flatten())

    def set_params(self, *, params: Params) -> None:
        self._Xtrain = params['exemplars']
        self._coeffs = params['coeffs']

    def get_params(self) -> Params:
        return Params(exemplars=self._Xtrain, coeffs=self._coeffs)
