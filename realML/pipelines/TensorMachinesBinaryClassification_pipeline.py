# pipeline example modified from David Johnson of the Michigan SPIDER team's GRASTA low-rank example, originally at 
# https://raw.githubusercontent.com/dvdmjohnson/d3m_michigan_primitives/master/spider/pipelines/unsupervised_learning_grasta.py
# following the setup of the CMU AutonBox semi-supervised binary classification primitive

from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context

from realML.pipelines.base import BasePipeline
from realML.kernel import TensorMachinesBinaryClassification

from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import Common as ColumnParserPrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from d3m.primitives.data_cleaning.imputer import SKlearn as SimpleImputerPrimitive
from sklearn_wrap.SKLinearSVR import SKLinearSVR

from d3m import index

class TensorMachinesBinaryClassificationPipeline(BasePipeline):

    #specify one seed dataset on which this pipeline can operate

    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        dataset = 'uu4_SPECT'
        #dataset = 'SEMI_1053_jm1'
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
        
        # Step 1: Simple Profiler Column Role Annotation
        step_1 = meta_pipeline.PrimitiveStep(
            primitive=index.get_primitive("d3m.primitives.schema_discovery.profiler.Common")
        )
        step_1.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.0.produce",
        )
        step_1.add_output("produce")
        pipeline.add_step(step_1)        

        # Step 1: ColumnParser
        step_2 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_2.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.1.produce')
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Step 3: Extract Attributes 
        step_3 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_3.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.2.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(name='semantic_types', 
		argument_type=ArgumentType.VALUE, 
		data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
        pipeline.add_step(step_3)

    	# Step 4: Impute missing attributes
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = SimpleImputerPrimitive.metadata.query())
        step_4.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.3.produce')
        step_4.add_output('produce')
        pipeline.add_step(step_4)

    	# Step 5: Convert attributes to ndarray
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_5.add_argument(
		name = 'inputs',
		argument_type = ArgumentType.CONTAINER,
		data_reference = 'steps.4.produce'
        )
        step_5.add_output('produce')
        pipeline.add_step(step_5)

        # Step 6: Extract Targets
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_6.add_argument(name='inputs', 
		argument_type=ArgumentType.CONTAINER, 
		data_reference='steps.2.produce')
        step_6.add_output('produce')
        step_6.add_hyperparameter(name='semantic_types', 
		argument_type=ArgumentType.VALUE, 
		data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
        pipeline.add_step(step_6)

    	# Step 7: Transform targets into an ndarray
        step_7 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_7.add_argument(
		name = 'inputs',
		argument_type = ArgumentType.CONTAINER,
		data_reference = 'steps.6.produce'
        )
        step_7.add_output('produce')
        pipeline.add_step(step_7)

        # Step 8: use TensorMachinesBinaryClassification
        step_8 = meta_pipeline.PrimitiveStep(primitive_description = TensorMachinesBinaryClassification.metadata.query())
        step_8.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.5.produce' #inputs here are the attributes from step 4
        )
        step_8.add_argument(
            name = 'outputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.7.produce' #outputs are the targets from step 6
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)
        
        #step 9: convert numpy-formatted prediction outputs to a dataframe
        step_9 = meta_pipeline.PrimitiveStep(primitive_description = NDArrayToDataFramePrimitive.metadata.query())
        step_9.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.8.produce'
        )
        step_9.add_output('produce')
        pipeline.add_step(step_9)

        # Step 10: generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
        step_10 = meta_pipeline.PrimitiveStep(primitive_description=ConstructPredictionsPrimitive.metadata.query())
        step_10.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.9.produce'  # inputs here are the prediction column
        )
        step_10.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.1.produce'  # inputs here are the dataframed input dataset
        )
        step_10.add_output('produce')
        pipeline.add_step(step_10)

        # Adding output step to the pipeline
        pipeline.add_output(
            name='output', 
            data_reference='steps.10.produce')

        return pipeline

if __name__ == '__main__':
        instance = TensorMachinesBinaryClassificationPipeline()
        json_info = instance.get_json()
        instanceid = instance.get_id()
        instancepath = os.path.join(".", instanceid)
        with open(instancepath + ".json", 'w') as file:
                file.write(json_info)
                file.close()
