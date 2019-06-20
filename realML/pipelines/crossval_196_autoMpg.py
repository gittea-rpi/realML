### CROSS-VALIDATION to select hyperparameters for the 196_autoMpg dataset
# uses the same preprocessing for all the primitives, which is not necessarily the best thing to do

# Import basics
import numpy as np
import os

# import D3M primitives and datatypes

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.container.numpy import ndarray as d3m_ndarray

from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFrame
from common_primitives.dataset_to_dataframe import Hyperparams as DatasetToDataFrameHyperparams

from common_primitives.column_parser import ColumnParserPrimitive as ColumnParser
from common_primitives.column_parser import Hyperparams as ColumnParserHyperparams

from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive as ExtractColumnsBySemanticTypes
from common_primitives.extract_columns_semantic_types import Hyperparams as ExtractColumnsHyperparams

from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive as DataFrameToNDArray
from common_primitives.dataframe_to_ndarray import Hyperparams as DataFrameToNDArrayHyperparams

from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive as NDArrayToDataFrame
from common_primitives.ndarray_to_dataframe import Hyperparams as NDArrayToDataFrameHyperparams

from d3m.primitives.data_transformation.encoder import DistilBinaryEncoder as BinaryEncoderPrimitive
from d3m.primitives.data_cleaning.imputer import SKlearn as ImputerPrimitive

# import realML primitives

from realML.kernel import RFMPreconditionedGaussianKRR, RFMPreconditionedPolynomialKRR
from realML.kernel import TensorMachinesRegularizedLeastSquares

# import SKLearn estimation and crossvalidation classes

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

import warnings
warnings.filterwarnings('ignore')


### load training data and preprocess it uniformly for all primitives
def load_training_data(datasetfname):
  dataset = Dataset.load(f'file://{datasetfname}')
  dataframe = DatasetToDataFrame(hyperparams=DatasetToDataFrameHyperparams.defaults()).produce(inputs=dataset).value
  dataframe = ColumnParser(hyperparams=ColumnParserHyperparams.defaults()).produce(inputs=dataframe).value

  ## extract the attributes
  attributes = ExtractColumnsBySemanticTypes(hyperparams=ExtractColumnsHyperparams.defaults()).produce(inputs=dataframe).value

  # impute missing values
  ImputerHyperparams = ImputerPrimitive.metadata.get_hyperparams().defaults().replace({'use_semantic_types':True})
  imputer = ImputerPrimitive(hyperparams=ImputerHyperparams)
  imputer.set_training_data(inputs=attributes)
  imputer.fit()
  attributes = imputer.produce(inputs=attributes).value
 
  # encode categorical values
  BinaryEncoderHyperparams = BinaryEncoderPrimitive.metadata.get_hyperparams().defaults().replace({'min_binary': 2})
  binaryencoder = BinaryEncoderPrimitive(hyperparams=BinaryEncoderHyperparams)
  binaryencoder.set_training_data(inputs=attributes)
  binaryencoder.fit()
  attributes_array = binaryencoder.produce(inputs=attributes).value
  attributes_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=attributes_array).value

  ## extract the targets
  targethyperparams = ExtractColumnsHyperparams.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']})
  targets = ExtractColumnsBySemanticTypes(hyperparams=targethyperparams).produce(inputs=dataframe).value
  targets_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=targets).value

  return attributes_array, targets_array

### create wrapper sklearn estimator classes so can use cross_validate
class gaussKRREstimator(BaseEstimator, RegressorMixin):
    def __init__(self, lparam=1, sigma=1):
        hyperparams = RFMPreconditionedGaussianKRR.metadata.get_hyperparams().defaults().replace({
            'lparam':lparam, 
            'sigma':sigma})
        self.fastEstimator = RFMPreconditionedGaussianKRR(hyperparams=hyperparams)
    
    def fit(self, X, y):
        self.fastEstimator.set_training_data(inputs=X, outputs=y)
        self.fastEstimator.fit()
    
    def predict(self, X):
        return self.fastEstimator.produce(inputs=X).value
    
    def get_params(self, deep=True):
        return {'lparam': self.fastEstimator.hyperparams['lparam'], 
                'sigma': self.fastEstimator.hyperparams['sigma']}

    def set_params(self, **parameters):
        curhyperparams = self.fastEstimator.hyperparams
        for parameter, value in parameters.items():
            curhyperparams = curhyperparams.replace({parameter: value})
        self.fastEstimator.hyperparams = curhyperparams
        return self
    
class polyKRREstimator(BaseEstimator, RegressorMixin):
    def __init__(self, lparam=1, offset = 0.001, sf = 1):
        hyperparams = RFMPreconditionedPolynomialKRR.metadata.get_hyperparams().defaults().replace({
            'lparam':lparam, 
            'offset':0.001, 
            'sf':1})
        self.fastEstimator = RFMPreconditionedPolynomialKRR(hyperparams=hyperparams)
    
    def fit(self, X, y):
        self.fastEstimator.set_training_data(inputs=X, outputs=y)
        self.fastEstimator.fit()
    
    def predict(self, X):
        return self.fastEstimator.produce(inputs=X).value
    
    def get_params(self, deep=True):
        return {'lparam': self.fastEstimator.hyperparams['lparam'], 
                'offset': self.fastEstimator.hyperparams['offset'],
                'sf': self.fastEstimator.hyperparams['sf']}

    def set_params(self, **parameters):
        curhyperparams = self.fastEstimator.hyperparams
        for parameter, value in parameters.items():
            curhyperparams = curhyperparams.replace({parameter: value})
        self.fastEstimator.hyperparams = curhyperparams
        return self

class TMRLSEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, q=3, r = 4, gamma = 0.1, alpha = 0.1, offset = 0.05, sf = .01):
        hyperparams = TensorMachinesRegularizedLeastSquares.metadata.get_hyperparams().defaults().replace({
            'q':q, 
            'r':r,
            'gamma':gamma,
            'alpha':alpha})
        self.fastEstimator = TensorMachinesRegularizedLeastSquares(hyperparams=hyperparams)
    
    def fit(self, X, y):
        self.fastEstimator.set_training_data(inputs=X, outputs=y)
        self.fastEstimator.fit()
    
    def predict(self, X):
        return self.fastEstimator.produce(inputs=X).value
    
    def get_params(self, deep=True):
        return {'q': self.fastEstimator.hyperparams['q'], 
                'r': self.fastEstimator.hyperparams['r'],
                'gamma': self.fastEstimator.hyperparams['gamma'],
                'alpha': self.fastEstimator.hyperparams['alpha']}

    def set_params(self, **parameters):
        curhyperparams = self.fastEstimator.hyperparams
        for parameter, value in parameters.items():
            curhyperparams = curhyperparams.replace({parameter: value})
        self.fastEstimator.hyperparams = curhyperparams
        return self
# Load the data and cross-validate to select hyperparameters for each primitive

datasetfname = os.path.abspath('/root/datasets/seed_datasets_current/196_autoMpg/196_autoMpg_dataset/datasetDoc.json')
X_train, y_train = load_training_data(datasetfname)
print("Cross-validating RFM Regression model parameters on dataset " + datasetfname)
print("Preprocessing: impute missing values, binary encode categorical values")

# generate the CV splits for this dataset
n_splits = 5
test_size = 0.25
random_state = 0
#cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
cv = KFold(n_splits=n_splits, shuffle=True, random_state = random_state)
print('Using ' + str(n_splits) + '-fold CV, same splits for each primitive')
print('NB: dataset has ' + str(X_train.shape[0]) + ' training points')

# set the error metric for this dataset
RMSE = lambda yT, yP: np.sqrt(mean_squared_error(yT, yP))

# Do cross-validation to get best Gaussian hyperparameters

print('trying model: Gaussian kernel ridge...')
kr = GridSearchCV(
    gaussKRREstimator(lparam=1.0, sigma=0.1), 
    cv=cv,
    param_grid={"lparam": [1e0, 0.1, 1e-2, 1e-3],
                "sigma": np.logspace(-2, 2.99, 7)},
    scoring=make_scorer(RMSE, greater_is_better=False)
)
kr.fit(X_train, y_train)
score = kr.best_score_ # that score is negative MSE scores. The thing is that GridSearchCV, by convention, always tries to maximize its score so loss functions like MSE have to be negated.
score = score*-1
print('model performance using CV (mean rmse)', score)
print(kr.best_estimator_)

# Do cross-validation to get the best polynomial hyperparameters
print('trying model: Polynomial kernel ridge...')
kr = GridSearchCV(
    polyKRREstimator(lparam=1.0, offset=0.1, sf=1.0), 
    cv=cv,
    param_grid={"lparam": [1e0, 0.1, 1e-2],
                "offset": np.logspace(-2, 0.3, 4),
                "sf": np.logspace(-2,0.3,4)},
    scoring=make_scorer(RMSE, greater_is_better=False)
)
kr.fit(X_train, y_train)
score = kr.best_score_ # that score is negative MSE scores. The thing is that GridSearchCV, by convention, always tries to maximize its score so loss functions like MSE have to be negated.
score = score*-1
print('model performance using CV (mean rmse)', score)
print(kr.best_estimator_)

# Do cross-validation to get the best tensor machines regularized least squares hyperparameters
print('trying model: Tensor Machines Regularized Least Squares...')
kr = GridSearchCV(
    TMRLSEstimator(q=3, r=4, gamma=0.1, alpha=0.01),
    cv=cv,
    param_grid={"q": [2, 3, 4],
                "r": [2, 4, 6, 8, 10],
                "gamma": np.logspace(-3, 0.9, 4),
                "alpha": np.logspace(-3, -0.001, 4)},
    scoring=make_scorer(RMSE, greater_is_better=False)
)
kr.fit(X_train, y_train)
score = kr.best_score_ # that score is negative MSE scores. The thing is that GridSearchCV, by convention, always tries to maximize its score so loss functions like MSE have to be negated.
score = score*-1
print('model performance using CV (mean rmse)', score)
print(kr.best_estimator_)
