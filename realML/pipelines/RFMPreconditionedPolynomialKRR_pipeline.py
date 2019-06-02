# pipeline example modified from David Johnson of the Michigan SPIDER team's regression example, originally at
# https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/supervised_learning_owl.py
from d3m.metadata import pipeline as d3m_pipeline
from d3m.metadata import base as d3m_base

import realML.pipelines.datasets
from realML.pipelines.base import BasePipeline
from realML.kernel import RFMPreconditionedPolynomialKRR
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive
from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
import os.path


class RFMPreconditionedPolynomialKRRPipeline(BasePipeline):
    def __init__(self):
        super().__init__()

        #specify one seed dataset on which this pipeline can operate
        dataset = 'LL0_207_autoPrice'
        self.meta_info = {
                'problem': realML.pipelines.datasets.get_problem_id(dataset),
		'full_inputs': [ realML.pipelines.datasets.get_full_id(dataset) ],
                'train_inputs': [ realML.pipelines.datasets.get_train_id(dataset) ],
                'test_inputs': [ realML.pipelines.datasets.get_problem_id(dataset) ],
		'score_inputs': [ realML.pipelines.datasets.get_score_id(dataset) ],
            }

    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = d3m_pipeline.Pipeline(context = d3m_base.Context.TESTING)
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')

        #step 0: Denormalize: join multiple tabular resource?
        # Why is there no entry point for Denormalize?

        #step 0: Dataset -> Dataframe
        step_0 = d3m_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        #step 1: ColumnParser
        step_1 = d3m_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_1.add_argument(
                name='inputs',
                argument_type=d3m_base.ArgumentType.CONTAINER,
                data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        #step 2: Extract attributes from dataset into a dedicated dataframe
        step_2 = d3m_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_2.add_argument(
                name = 'inputs',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(
                name='semantic_types',
                argument_type=d3m_base.ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline.add_step(step_2)

        #step 3: Extract Targets
        step_3 = d3m_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_3.add_argument(name='inputs', argument_type=d3m_base.ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(
                name='semantic_types',
                argument_type=d3m_base.ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        pipeline.add_step(step_3)

        #step 4: transform targets dataframe into an ndarray
        step_4 = d3m_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_4.add_argument(
                name = 'inputs',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'steps.3.produce'
        )
        step_4.add_output('produce')
        pipeline.add_step(step_4)

        #step 5 : transform features dataframe into an ndarray
        step_5 = d3m_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_5.add_argument(
            name = 'inputs',
            argument_type = d3m_base.ArgumentType.CONTAINER,
            data_reference = 'steps.2.produce'
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)
        attributes = 'steps.5.produce'
        targets    = 'steps.4.produce'

        #step 6: call RFMPreconditionedPolynomialKRR for regression
        step_6 = d3m_pipeline.PrimitiveStep(primitive_description=RFMPreconditionedPolynomialKRR.metadata.query())
        step_6.add_argument(
                name='inputs',
                argument_type=d3m_base.ArgumentType.CONTAINER,
                data_reference=attributes)
        step_6.add_argument(
                name='outputs',
                argument_type=d3m_base.ArgumentType.CONTAINER,
                data_reference=targets)
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        #step 7: convert numpy-formatted prediction outputs to a dataframe
        step_7 = d3m_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_7.add_argument(
                name = 'inputs',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'steps.6.produce'
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        #step 8: generate a properly-formatted output dataframe from the dataframed prediction outputs using the input dataframe as a reference
        step_8 = d3m_pipeline.PrimitiveStep(primitive_description = ConstructPredictionsPrimitive.metadata.query())
        step_8.add_argument(
                name = 'inputs',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'steps.7.produce' #inputs here are the prediction column
        )
        step_8.add_argument(
                name = 'reference',
                argument_type = d3m_base.ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce' #inputs here are the dataframe input dataset
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)

        # Final Output
        pipeline.add_output(
                name='output',
                data_reference='steps.8.produce')

        return pipeline

if __name__ == '__main__':
	instance = RFMPreconditionedPolynomialKRRPipeline()
	json_info = instance.get_json()
	instanceid = instance.get_id()
	instancepath = os.path.join(".", instanceid)
	with open(instancepath + ".json", 'w') as file:
		file.write(json_info)
		file.close()
