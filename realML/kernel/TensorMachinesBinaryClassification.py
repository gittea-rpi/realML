import typing
from typing import Any, List, Dict, Union, Optional, Sequence
from collections import OrderedDict
from numpy import ndarray, sign
import numpy as np
import os
from .tensormachines import tm_fit, tm_predict, tm_preprocess

from d3m import utils
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    weights : Optional[ndarray]
    norms : Optional[ndarray]

class Hyperparams(hyperparams.Hyperparams):
    # search over these hyperparameters to tune performance
    q = hyperparams.UniformInt(default=3, lower=2, upper=10, 
                               description="degree of the polynomial to be fit", 
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    r = hyperparams.UniformInt(default=5, lower=2, upper=30, 
                               description="rank of the coefficient tensors to be fit", 
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    gamma = hyperparams.LogUniform(default=.01, lower=.0001, upper=10, 
                                   description="l2 regularization to use on the tensor low-rank factors", 
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.LogUniform(default=.1, lower=.001, upper=1, 
                                   description="variance of the random initialization of the factors", 
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    epochs = hyperparams.UniformInt(default=30, lower=1, upper=100, 
                                    description="maximum iterations of LBFGS, or number of epochs of SFO", 
                                    semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    # control parameters determined once during pipeline building then fixed
    solver = hyperparams.Enumeration[str](default="LBFGS", values=["SFO", "LBFGS"], 
                             description="solver to use: LBFGS better for small enough datasets, SFO does minibached stochastic quasi-Newton to scale to large dataset", 
                             semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
    preprocess = hyperparams.Enumeration[str](default="YES", values=["YES", "NO"], 
                             description="whether to use a preprocessing that tends to work well for tensor machines", 
                             semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])

class TensorMachinesBinaryClassification(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Learns a polynomial function using logistic regression for binary classification by modeling the polynomial's coefficients as low-rank tensors.
    Meant as a faster, more scalable alternative to polynomial random feature map approaches like CRAFTMaps.
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'f8de43e0-7f81-4edd-9ef6-51bcd2953784',
        'version': __version__,
        'name': 'Tensor Machine Binary Classifier',
        'description': 'Fit a polynomial function for logistic regression by modeling the polynomial coefficients as collection of low-rank tensors',
        'python_path': 'd3m.primitives.classification.tensor_machines_binary_classification.TensorMachinesBinaryClassification',
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        'algorithm_types' : [
            'KERNEL_METHOD',
            'LOGISTIC_REGRESSION',
            'POLYNOMIAL_NEURAL_NETWORK'
        ],
        'keywords' : ['kernel learning', 'binary classification', 'adaptive features', 'polynomial model', 'classification'],
        'source' : {
            'name': __author__,
            'contact': 'mailto:gittea@rpi.edu',
            'citation': 'https://arxiv.org/abs/1504.01697',
            'uris' : [
                "https://github.com/ICSI-RealML/realML.git",
            ],
        },
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/ICSI-RealML/realML.git@{git_commit}#egg=realML'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ],
        'location_uris': [ # NEED TO REF SPECIFIC COMMIT
            'https://github.com/ICSI-RealML/realML/blob/master/realML/kernel/TensorMachinesBinaryClassification.py',
            ],
        'preconditions': [
            'NO_MISSING_VALUES',
            'NO_CATEGORICAL_VALUES'
        ],
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._seed = random_seed
        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False
        self._weights = None
        self._norms = None

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs

        if self.hyperparams['preprocess'] == 'YES':
            (self._training_inputs, self._norms) = tm_preprocess(self._training_inputs)

        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        if len(self._training_outputs.shape) == 1:
            self._training_outputs = np.expand_dims(self._training_outputs, axis=1)
        (self._weights, _) = tm_fit(self._training_inputs, self._training_outputs, 'bc', self.hyperparams['r'],
           self.hyperparams['q'], self.hyperparams['gamma'], self.hyperparams['solver'],
           self.hyperparams['epochs'], self.hyperparams['alpha'], seed=self._seed)

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self.hyperparams['preprocess'] == 'YES':
            inputs = tm_preprocess(inputs, colnorms=self._norms)

        pred_test = tm_predict(self._weights, inputs, self.hyperparams['q'],
                                     self.hyperparams['r'], 'bc')
        return CallResult(sign(pred_test.flatten()).astype(int))

    def get_params(self) -> Params:
        return Params(weights=self._weights, norms=self._norms)

    def set_params(self, *, params: Params) -> None:
        self._weights = params['weights']
        self._norms = params['norms']
