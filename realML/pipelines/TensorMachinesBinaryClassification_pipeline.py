# pipeline example modified from David Johnson of the Michigan SPIDER team's GRASTA low-rank example, originally at 
# https://raw.githubusercontent.com/dvdmjohnson/d3m_michigan_primitives/master/spider/pipelines/unsupervised_learning_grasta.py
# following the setup of the CMU AutonBox semi-supervised binary classification primitive

from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from realML.pipelines.base import BasePipeline
from realML.kernel import TensorMachinesBinaryClassification

from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import DataFrameCommon as ColumnParserPrimitive
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_cleaning.imputer import SKlearn as SimpleImputerPrimitive
from sklearn_wrap.SKLinearSVR import SKLinearSVR


class TensorMachinesBinaryClassificationPipeline(BasePipeline):

    #specify one seed dataset on which this pipeline can operate

    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        dataset = 'SEMI_1053_jm1'
        self.meta_info = self.genmeta(dataset)

    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')

        # Step 0: DatasetToDataFrame
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(name='inputs', 
		argument_type = ArgumentType.CONTAINER, 
		data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Step 1: ColumnParser
        step_1 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_1.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Step 2: Extract Attributes 
        step_2 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_2.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(name='semantic_types', 
		argument_type=ArgumentType.VALUE, 
		data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
        pipeline.add_step(step_2)

	# Step 3: Impute missing data
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = SimpleImputerPrimitive.metadata.query())
        step_3.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.2.produce')
        step_3.add_output('produce')
        pipeline.add_step(step_3)

        # Step 4: Extract Targets
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_4.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.1.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(name='semantic_types', 
		argument_type=ArgumentType.VALUE, 
		data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
        pipeline.add_step(step_4)

	# Step 5: Transform targets into an ndarray
	step_5 = d3m_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
	step_5.add_argument(
		name = 'inputs',
		argument_type = d3m_base.ArgumentType.CONTAINER,
		data_reference = 'steps.4.produce'
	)
	pipeline.add_step(step_5)

        # Step 6: use TensorMachinesBinaryClassification
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = TensorMachinesBinaryClassification.metadata.query())
        step_6.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.3.produce' #inputs here are the attributes from step 3
        )
        step_6.add_argument(
            name = 'outputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.5.produce' #outputs are the targets from step 5
	)
        step_6.add_output('produce')
        pipeline.add_step(step_6)
        
        # Step 7: generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_7 = meta_pipeline.PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
        step_7.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.6.produce'  # inputs here are the prediction column
        )
        step_7.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'  # inputs here are the dataframed input dataset
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        # Adding output step to the pipeline
        pipeline.add_output(
            name='output', 
            data_reference='steps.7.produce')

        return pipeline

if __name__ == '__main__':
        instance = TensorMachinesBinaryClassificationPipeline()
        json_info = instance.get_json()
        instanceid = instance.get_id()
        instancepath = os.path.join(".", instanceid)
        with open(instancepath + ".json", 'w') as file:
                file.write(json_info)
                file.close()
