import os
from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFrame
from common_primitives.dataset_to_dataframe import Hyperparams as DatasetToDataFrameHyperparams

dataset_doc_path = os.path.abspath('/root/datasets/seed_datasets_current/uu4_SPECT/uu4_SPECT_dataset/datasetDoc.json')
dataset = Dataset.load(f'file://{dataset_doc_path}')
dataframe = DatasetToDataFrame(hyperparams=DatasetToDataFrameHyperparams.defaults()).produce(inputs=dataset).value
