from typing import Optional
import os

import numpy as np  # type: ignore
from scipy import linalg  # type: ignore
import scipy as sci  # type: ignore

from d3m.container import ndarray
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.metadata import base as metadata_base, hyperparams, params
import d3m.metadata.base as metadata_module
from d3m import exceptions, utils

from . import __author__, __version__

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer


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
    
    
    degree = hyperparams.Hyperparameter[int](
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=2,
        description="The degree of the polynomial features. Default = 2.",
    )    


class RandomizedPolyPCA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Given a mean rectangular matrix `A` with shape `(m, n)`, a set of polynomial features of degree n
    is constructed. Then the randomized PCA is used to extract a new set of components 
    that captures most of the variation in the data.
    """

    __author__ = "ICSI" # a la directions on https://gitlab.datadrivendiscovery.org/jpl/primitives_repo
    metadata = metadata_base.PrimitiveMetadata({
        'id': '2b39791f-03aa-41ea-b370-abdd043a8887',
        'version': __version__,
        'name': 'Randomized Principal Component Analysis using Polynomial Features',
        'description': "Extract the dominant PCA modes from polynomial and interaction features.",
        'python_path': 'd3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA',
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        'algorithm_types' : [
            'LOW_RANK_MATRIX_APPROXIMATIONS'
        ],
        'keywords' : ['low rank approximation', 'randomized PCA'],
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
            'https://github.com/ICSI-RealML/realML/blob/master/realML/matrix/randomizedpolypca.py',
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

        # Do some preprocessing to pass CI
        #self._training_inputs = np.array(self._training_inputs)
        #self._training_inputs[np.isnan(self._training_inputs)] = 0
        
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(self._training_inputs)
        self._training_inputs = imp_mean.transform(self._training_inputs)        
        
        
        # Create features
        poly = PolynomialFeatures(degree=self.hyperparams['degree'], interaction_only=False)
        X = poly.fit_transform(self._training_inputs)
        #poly = PolynomialFeatures(interaction_only=True)
        #X = poly.fit_transform(X)

        
        # Center data
        self._mean = X.mean(axis=0)
        X = X - self._mean
        
        
        # Shape of input matrix 
        m , n = X.shape
        k = self.hyperparams['n_components']
        q = 3
        p = 20
            
        if k > min(m,n):
            k = min(m,n)       
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Generate a random test matrix Omega
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Omega = np.random.standard_normal(size=(n, k+p)) 
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Build sample matrix Y : Y = A * Omega
        #Note: Y should approximate the range of A
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        Y = X.dot(Omega)
        del(Omega)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Orthogonalize Y using economic QR decomposition: Y=QR
        #If q > 0 perfrom q subspace iterations
        #Note: check_finite=False may give a performance gain
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
        s = 1 #control parameter for number of orthogonalizations
        if q > 0:
            for i in np.arange( 1, q+1 ):
                if( (2*i-2) % s == 0 ):
                    Y , _ = sci.linalg.qr(Y , mode='economic', check_finite=False, overwrite_a=True)
                            
                if( (2*i-1) % s == 0 ):
                    Z , _ = sci.linalg.qr(X.T.dot(Y), mode='economic', check_finite=False, overwrite_a=True)
           
                Y = X.dot(Z)
            #End for
         #End if       
            
        Q , _ = sci.linalg.qr(Y ,  mode='economic', check_finite=False, overwrite_a=True)  
        del(Y)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Project the data matrix a into a lower dimensional subspace
        #B = Q.T * A 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        B = Q.T.dot(X)   
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Singular Value Decomposition
        #Note: B = U" * S * Vt
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
        #Compute SVD
        U , s , Vt = sci.linalg.svd(B ,  compute_uv=True,
                                  full_matrices=False, 
                                  overwrite_a=True,
                                  check_finite=False)
         
        #Recover right singular vectors
        #U = Q.dot(U)
    
        #Return Trunc
        Vt =  Vt[0:k,:]

        # Construct transformation matrix with eigenvectors
        self._invtransformation = Vt
        self._transformation = Vt.T

        self._fitted = True
        return CallResult(None)    

    #**************************************************************************   
    #End rsvd
    #**************************************************************************     
        
        

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        "Returns the latent matrix"
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")
            
        # Do some preprocessing to pass CI
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(inputs)
        inputs = imp_mean.transform(inputs) 

        imp_question = SimpleImputer(missing_values='?', strategy='mean')
        imp_question.fit(inputs)
        inputs = imp_question.transform(inputs)                     
            
        # Create features
        poly = PolynomialFeatures(degree=self.hyperparams['degree'], interaction_only=False)
        X = poly.fit_transform(inputs)
        #poly = PolynomialFeatures(interaction_only=True)
        #X = poly.fit_transform(X)            
            
        comps = (X - self._mean).dot(self._transformation)
        
        # remove nan
        imp_mean2 = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean2.fit(comps)
        comps = imp_mean2.transform(comps)         
        
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