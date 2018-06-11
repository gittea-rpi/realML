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

from .approxL1LowRankDecomposition import *

from d3m import utils
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult, DockerContainer

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    A: Optional[ndarray]
    B: Optional[ndarray]

class Hyperparams(hyperparams.Hyperparams):
    # search over these hyperparameters to tune performance
    rank = hyperparams.UniformInt(default=5, lower=1, upper=500,
                                  description="desired rank of the decomposition",
                                  semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # control parameters determined once during pipeline building then fixed
    rankMultiplier = hyperparams.UniformInt(default=5, lower=3, upper=12, 
                                      description="work in dimension that is this multiple of the desired final rank", 
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    numReps = hyperparams.UniformInt(default=1, lower=1, upper=20, 
                                      description="repeat the approximation this many times and take the best approximation",
                                      semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

class L1LowRank(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Performs fast approximate solution to the NP-hard problem of computing a low-rank approximation that minimizes
    the entrywise l1-norm approximation error
        (A,B) = argmin ||X -  A B||_1
    subject to A and B.T have a fixed number of columns k
    The algorithm is principled, and has an approximation factor that grows like poly(k*log(n)), when X is n-by-n; it
    meets this guarantee with constant probability.
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'ea3b78a6-dc8c-4772-a329-b653583817b4',
        'version': __version__,
        'name': 'Fast Approximate Entrywise L1-Norm Low Rank Factorization',
        'description': 'Finds a low-rank approximation that approximately minimizes the sum of the entry-wise absolute value of the error matrix',
        'python_path': 'd3m.primitives.realML.L1LowRank',
        'primitive_family': 'FEATURE_EXTRACTION',
        'algorithm_types' : [
            'LOW_RANK_MATRIX_APPROXIMATIONS'
        ],
        'keywords' : ['low rank approximation', 'robust PCA'],
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
            'https://github.com/ICSI-RealML/realML/blob/master/realML/matrix/L1LowRank.py',
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
        Initializes the L1LowRank solver
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._rank = self.hyperparams['rank']
        self._seed = random_seed
        self._featMat = None
        self._fitted = False
        np.random.seed(random_seed)
    
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Sets the training data:
            Input: array, shape = [n_samples, n_features]
        """
        self._featMat = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Learns the LAD regression coefficients alpha given training pairs (X,y)
        """
        if self._fitted:
            return CallResult(None)

        if self._featMat is None:
            raise ValueError("Missing training data.")

        rankMultiplier = self.hyperparams['rankMultiplier']
        numReps = self.hyperparams['numReps']
        self._A, self._B = FastL1LowRank(self._featMat, self._rank, rankMultiplier, numReps)

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Project each row of input onto the row space of the low rank approximation to the training data

        Inputs:
            X: array of shape [n_samples, n_features]
        Outputs:
            y: array of shape [n_samples, n_targets]
        """
        Q = orth(self._B.T)
        return CallResult(inputs.dot(Q).dot(Q.T))

    def set_params(self, *, params: Params) -> None:
        self._A = params['A']
        self._B = params['B']

    def get_params(self) -> Params:
        return Params(A=self._A, B=self._B)
