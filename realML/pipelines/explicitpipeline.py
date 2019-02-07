import os

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
print('importing common primitives package...')
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFrame
from common_primitives.dataset_to_dataframe import Hyperparams as DatasetToDataFrameHyperparams
from common_primitives.column_parser import ColumnParserPrimitive as ColumnParser
from common_primitives.column_parser import Hyperparams as ColumnParserHyperparams
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive as ExtractColumnsBySemanticTypes
from common_primitives.extract_columns_semantic_types import Hyperparams as ExtractColumnsHyperparams
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive as DataFrameToNDArray
from common_primitives.dataframe_to_ndarray import Hyperparams as DataFrameToNDArrayHyperparams
from realML.matrix.FastLADSolver import FastLAD
from realML.matrix.FastLADSolver import Hyperparams as FastLADHyperparams
from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive as NDArrayToDataFrame
from common_primitives.ndarray_to_dataframe import Hyperparams as NDArrayToDataFrameHyperparams

print('done')

dataset_doc_path = os.path.abspath('/root/datain/LL0_207_autoPrice/LL0_207_autoPrice_dataset/datasetDoc.json')
dataset = Dataset.load(f'file://{dataset_doc_path}')
dataframe = DatasetToDataFrame(hyperparams=DatasetToDataFrameHyperparams.defaults()).produce(inputs=dataset).value
dataframe = ColumnParser(hyperparams=ColumnParserHyperparams.defaults()).produce(inputs=dataframe).value
attributes = ExtractColumnsBySemanticTypes(hyperparams=ExtractColumnsHyperparams.defaults()).produce(inputs=dataframe).value
attributes_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=attributes).value
targethyperparams = ExtractColumnsHyperparams.defaults().replace({'semantic_types':['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']})
targets = ExtractColumnsBySemanticTypes(hyperparams=targethyperparams).produce(inputs=dataframe).value
targets_array = DataFrameToNDArray(hyperparams=DataFrameToNDArrayHyperparams.defaults()).produce(inputs=targets).value
predictor = FastLAD(hyperparams=FastLADHyperparams.defaults())
predictor.set_training_data(inputs=attributes_array,outputs=targets_array)
predictor.fit()
predictions = predictor.produce(inputs=attributes_array).value
predictionsdf = NDArrayToDataFrame(hyperparams=NDArrayToDataFrameHyperparams.defaults()).produce(inputs=predictions).value
print(predictionsdf)



