import typing
from typing import Any, List, Dict, Union, Optional, Sequence
from collections import OrderedDict
from numpy import ndarray
import os

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
import scipy.sparse.linalg
import numpy.linalg
from scipy.linalg import norm

from .LADregression import *

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
    # control parameters determined once during pipeline building then fixed
    coresetmultiplier = hyperparams.UniformInt(default=4, lower=2, upper=7, 
                                      description="coreset size, as a multiple of the number of input features", 
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    eps = hyperparams.LogUniform(default=1e-6, lower=1e-14, upper=1e-2, 
                                 description="relative error stopping tolerance for IRLS solver", 
                                 semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    maxIters = hyperparams.UniformInt(default=100, lower=50, upper=500, 
                                      description="maximum iterations of IRLS", 
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])


# TO DO: normalize the objective so lparam and sigma don't need a data-dependent range!
class FastLAD(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Performs fast least absolute deviations regression by forming a coreset and solving LAD on that coreset using IRLS
    to return an approximate solution alphahat to 
        alpha = argmin ||A alpha - y||_1
    predictions are then formed by 
        ypred = trainingData * alphahat
    For details see Magdon-Ismail, Gittens 2018 
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'b158a49d-5deb-462e-b7e3-e321624dad89',
        'version': __version__,
        'name': 'Coreset-based Fast Least Absolute Deviations Solver',
        'description': 'Fast solver for least absolute deviations regression, using a coreset',
        'python_path': 'd3m.primitives.realML.FastLAD',
        'primitive_family': 'REGRESSION',
        'algorithm_types' : [
            'MULTIVARIATE_REGRESSION'
        ],
        'keywords' : ['least absolute deviations', 'regression', 'linear regression', 'robust', 'robust regression'],
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
            'https://github.com/ICSI-RealML/realML/blob/master/realML/matrix/FastLADSolver.py',
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
        Initializes the LAD solver
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

        if len(self._ytrain.shape) == 1:
            self._ytrain = np.expand_dims(self._ytrain, axis=1) 

        self._n, self._d = self._Xtrain.shape
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Learns the LAD regression coefficients alpha given training pairs (X,y)
        """
        if self._fitted:
            return CallResult(None)

        if self._Xtrain is None or self._ytrain is None:
            raise ValueError("Missing training data.")

        stoppingTol = self.hyperparams['eps']*norm(self._ytrain, 1)/(np.sqrt(self._n)*norm(self._Xtrain))
        r = self.hyperparams['coresetmultiplier']*self._d

        if r < self._n :
            self._U = generateWellConditionedBasis(np.concatenate((self._Xtrain, self._ytrain), axis=1), r)
            self._coeffs = coresetLAD(self._Xtrain, self._ytrain, self._U, r,
                              stoppingTol, self.hyperparams['maxIters'])
        else:
            print("coreset size is larger than number of examples, so solving the full LAD problem --- you may want to lower the coresetmultiplier parameter")
            self._coeffs = LAD(self._Xtrain, self._ytrain, stoppingTol, self.hyperparams['maxIters'])

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
        return CallResult(inputs.dot(self._coeffs))

    def set_params(self, *, params: Params) -> None:
        self._Xtrain = params['exemplars']
        self._coeffs = params['coeffs']

    def get_params(self) -> Params:
        return Params(exemplars=self._Xtrain, coeffs=self._coeffs)
