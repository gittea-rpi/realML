import os
from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame

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

from realML.matrix.approxL1LowRankDecomposition import *

dataset_doc_path = os.path.abspath('/root/datasets/seed_datasets_current/uu4_SPECT/uu4_SPECT_dataset/datasetDoc.json')
dataset_doc_path = os.path.abspath('/root/datasets/seed_datasets_current/196_autoMpg/196_autoMpg_dataset/datasetDoc.json')
dataset = Dataset.load(f'file://{dataset_doc_path}')
dataframe = DatasetToDataFrame(hyperparams=DatasetToDataFrameHyperparams.defaults()).produce(inputs=dataset).value
dataframe = ColumnParser(hyperparams=ColumnParserHyperparams.defaults()).produce(inputs=dataframe).value

attributes = ExtractColumnsBySemanticTypes(hyperparams=ExtractColumnsHyperparams.defaults()).produce(inputs=dataframe).value
attributes_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=attributes).value

targethyperparams = ExtractColumnsHyperparams.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']})
targets = ExtractColumnsBySemanticTypes(hyperparams=targethyperparams).produce(inputs=dataframe).value
targets_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=targets).value

print(attributes_array)
bestA, bestB = L1LowRank(attributes_array, 5)
features_array = bestA.dot(bestB) 

predictor = SKLinearSVR(hyperparams=SKLinearSVRHyperparams.defaults())
predictors.set_training_data(inputs=features_array,outputs=targets_array)
predictor.fit()

predictions = predictor.produce(inputs=features_array).value
predictionsdf = NDArrayToDataFrame(hyperparams=NDArrayToDataFrameHyperparams.defaults()).produce(inputs=predictions).value
print(predictionsdf)


