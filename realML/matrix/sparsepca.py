from typing import Optional
import os

import numpy as np  # type: ignore
from scipy import linalg  # type: ignore

from d3m.container import ndarray
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base, hyperparams, params
import d3m.metadata.base as metadata_module
from d3m import exceptions, utils

from . import __author__, __version__


Inputs = ndarray
Outputs = ndarray


class Params(params.Params):
    transformation: Optional[np.ndarray]
    mean: Optional[np.ndarray]


class Hyperparams(hyperparams.Hyperparams):
    n_components = hyperparams.Hyperparameter[int](
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=1,
        description="Target rank, i.e., number of sparse components to be computed.",
    )
    max_iter = hyperparams.Hyperparameter[int](
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
        default=100,
        description="Maximum number of iterations to perform before exiting."
    )
    max_tol = hyperparams.Hyperparameter[float](
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
        default=1e-5,
        description="Stopping tolerance for reconstruction error."
    )

    # search over these hyperparameters to tune performance
    alpha = hyperparams.Uniform(
        default=1e-1, lower=0.0, upper=1.0,
        description="Sparsity controlling parameter. Higher values lead to sparser components",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    beta = hyperparams.Uniform(
        default=1e-6, lower=0.0, upper=1e-1,
        description="Amount of ridge shrinkage to apply in order to improve conditionin.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )


class SparsePCA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA
    computes a set of sparse components that can optimally reconstruct the
    input data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha. In addition, some ridge
    shrinkage can be applied in order to improve conditioning.
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'ea3b78a6-dc8c-4772-a329-b653583817b4',
        'version': __version__,
        'name': 'Sparse Principal Component Analysis',
        'description': "Given a mean centered rectangular matrix `A` with shape `(m, n)`, SPCA computes a set of sparse components that can optimally reconstruct the input data.  The amount of sparseness is controllable by the coefficient of the L1 penalty, given by the parameter alpha. In addition, some ridge shrinkage can be applied in order to improve conditioning.",
        'python_path': 'd3m.primitives.feature_extraction.sparse_pca.SparsePCA',
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        'algorithm_types' : [
            'LOW_RANK_MATRIX_APPROXIMATIONS'
        ],
        'keywords' : ['low rank approximation', 'sparse PCA'],
        'source' : {
            'name': __author__,
            'contact': 'mailto:erichson@berkeley.edu',
            'uris' : [
                'https://github.com/ICSI-RealML/realML.git',
            ],
        },
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/ICSI-RealML/realML.git@{git_commit}#egg=realML'.format(git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ],
        'location_uris': [ # NEED TO REF SPECIFIC COMMIT
            'https://github.com/ICSI-RealML/realML/blob/master/realML/matrix/sparsepca.py',
            ],
        'preconditions': [
            'NO_MISSING_VALUES',
            'NO_CATEGORICAL_VALUES'
        ],
    })    
    
    
    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._fitted = False
        self._transformation = None
        self._mean = None
        # Used only for testing.
        self._invtransformation = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        # If already fitted with current training data, this call is a noop.
        if self._fitted:
            return CallResult(None)
        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # Center data
        self._mean = self._training_inputs.mean(axis=0)
        X = self._training_inputs - self._mean
        # Initialization of Variable Projection Solver
        U, D, Vt = linalg.svd(X, full_matrices=False, overwrite_a=False)
        Dmax = D[0]  # l2 norm
        A = Vt[:self.hyperparams['n_components']].T
        B = Vt[:self.hyperparams['n_components']].T
        VD = Vt.T * D
        VD2 = Vt.T * D**2
        # Set Tuning Parameters
        alpha = self.hyperparams['alpha']
        beta = self.hyperparams['beta']
        alpha *= Dmax**2
        beta *= Dmax**2
        nu = 1.0 / (Dmax**2 + beta)
        kappa = nu * alpha
        obj = []  # values of objective function
        n_iter = 0

        #   Apply Variable Projection Solver
        while self.hyperparams['max_iter'] > n_iter:
            # Update A:
            # X'XB = UDV'
            # Compute X'XB via SVD of X
            Z = VD2.dot(Vt.dot(B))
            Utilde, Dtilde, Vttilde = linalg.svd(Z, full_matrices=False, overwrite_a=True)
            A = Utilde.dot(Vttilde)
            # Proximal Gradient Descent to Update B
            G = VD2.dot(Vt.dot(A - B)) - beta * B
            arr = B + nu * G
            B = np.sign(arr) * np.maximum(np.abs(arr) - kappa, 0)
            # Compute residuals
            R = VD.T - VD.T.dot(B).dot(A.T)
            # Calculate objective
            obj.append(0.5*np.sum(R**2) + alpha*np.sum(np.abs(B)) + 0.5*beta*np.sum(B**2))
            # Break if obj is not improving anymore
            if n_iter > 0 and abs(obj[-2] - obj[-1]) / obj[-1] < self.hyperparams['max_tol']:
                break
            # Next iter
            n_iter += 1

        # Construct transformation matrix with eigenvectors
        self._invtransformation = A
        self._transformation = B

        self._fitted = True
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        "Returns the latent matrix"
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")
        comps = (inputs - self._mean).dot(self._transformation)
        return CallResult(ndarray(comps, generate_metadata=True))

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def get_params(self) -> Params:
        if self._fitted:
            return Params(
                transformation=self._transformation,
                mean=self._mean,
            )
        else:
            return Params(
                transformation=None,
                mean=None,
            )

    def set_params(self, *, params: Params) -> None:
        self._transformation = params['transformation']
        self._mean = params['mean']
        self._fitted = all(param is not None for param in params.values())