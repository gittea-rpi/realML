import typing
from typing import Any, List, Dict, Union, Optional, Sequence
from collections import OrderedDict
from numpy import ndarray
import os, sklearn
from .tensormachines import tm_fit, tm_predict, tm_preprocess

from d3m_metadata.container.numpy import ndarray as d3m_ndarray
from d3m_metadata import hyperparams, params, metadata as metadata_module, utils
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    weights : Optional[ndarray]
    norms : Optional[ndarray]

class Hyperparams(hyperparams.Hyperparams):
    # search over these hyperparameters to tune performance
    q = hyperparams.UniformInt(default=3, lower=2, upper=9, description="degree of the polynomial to be fit", 
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    r = hyperparams.UniformInt(default=5, lower=2, upper=10, description="rank of the coefficient tensors to be fit", 
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    gamma = hyperparams.LogUniform(default=.01, lower=.0001, upper=10, description="l2 regularization to use on the tensor low-rank factors", 
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    alpha = hyperparams.LogUniform(default=.1, lower=.001, upper=1, description="variance of the random initialization of the factors", 
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    epochs = hyperparams.UniformInt(default=30, lower=15, upper=100, description="maximum iterations of LBFGS, or number of epochs of SFO", 
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
    Primitive wrapping for sklearn ExtraTreesClassifier
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'ecc83605-d340-490d-9a2d-81c2ea6cb6cb', #uuid3(NAMESPACE_DNS, "realML.kernel.TensorMachineBinaryClassification" + __version__),
        'version': __version__,
        'name': 'Tensor Machine Binary Classifier',
        'description': 'Fit a polynomial function for logistic regression by modeling the polynomial coefficients as collection of low-rank tensors',
        'python_path': 'd3m.primitives.realML.kernel.TensorMachinesBinaryClassification',
        'primitive_family': metadata_module.PrimitiveFamily.CLASSIFICATION,
        'algorithm_types' : [
            metadata_module.PrimitiveAlgorithmType.LOGISTIC_REGRESSION,
        ],
        'keywords' : ['kernel learning', 'binary classification', 'adaptive features', 'polynomial model', 'classification'],
        'source' : {
            'name': __author__,
            'contact': 'mailto:gittea@rpi.edu',
            'citation': 'https://arxiv.org/abs/1504.01697',
            'uris' : [
                "http://https://github.com/alexgittens/realML.git",
            ],
        },
        'installation': [
            {
                'type': metadata_module.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/alexgittens/realML.git@{git_commit}#egg=realML'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ],
        'location_uris': [ # NEED TO REF SPECIFIC COMMIT
            'https://github.com/alexgittens/realML/blob/master/realML/kernel/TensorMachinesBinaryClassification.py',
            ],
        'preconditions': [
            metadata_module.PrimitivePrecondition.NO_MISSING_VALUES,
            metadata_module.PrimitivePrecondition.NO_CATEGORICAL_VALUES
        ],
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None) -> None:

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

        (self._weights, _) = tm_fit(self._training_inputs, self._training_outputs, 'bc', self.hyperparams['r'],
           self.hyperparams['q'], self.hyperparams['gamma'], self.hyperparams['solver'],
           self.hyperparams['epochs'], self.hyperparams['alpha'], seed=self._seed)

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self.hyperparams['preprocess'] == 'YES':
            inputs = tm_preprocess(inputs, colnorms=self._norms)

        return CallResult(self._clf.predict(inputs))

    def get_params(self) -> Params:
        return Params(weights=self._weights, norms=self._norms)

    def set_params(self, *, params: Params) -> None:
        self._weights = params['weights']
        self._norms = params['norms']
