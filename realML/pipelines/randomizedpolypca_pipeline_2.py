# pipeline example modified from David Johnson of the Michigan SPIDER team's GRASTA low-rank example, originally at 
# https://raw.githubusercontent.com/dvdmjohnson/d3m_michigan_primitives/master/spider/pipelines/unsupervised_learning_grasta.py

from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata import base as d3m_base



from realML.pipelines.base import BasePipeline
from realML.matrix import RandomizedPolyPCA

from d3m.primitives.data_transformation.dataframe_to_ndarray import Common as DataFrameToNDArrayPrimitive
from d3m.primitives.data_transformation.ndarray_to_dataframe import Common as NDArrayToDataFramePrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.column_parser import Common as ColumnParserPrimitive
from d3m.primitives.data_transformation.construct_predictions import Common as ConstructPredictionsPrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import Common as ExtractColumnsBySemanticTypesPrimitive
from sklearn_wrap.SKLinearSVR import SKLinearSVR

import d3m.primitives.classification.gradient_boosting
#
import d3m.primitives.regression.gradient_boosting

import d3m.primitives.data_cleaning.imputer

from d3m import index


import random

class randomizedpolypcaPipeline2(BasePipeline):

    #specify one seed dataset on which this pipeline can operate

    def __init__(self):
        super().__init__()
        
        #specify one seed dataset on which this pipeline can operate
        dataset = 'SEMI_1053_jm1_MIN_METADATA'
        self.meta_info = self.genmeta(dataset)
        random.seed(123)
        
    #define pipeline object
    def _gen_pipeline(self):
        #pipeline context is just metadata, ignore for now
        pipeline = meta_pipeline.Pipeline()
        #define inputs.  This will be read in automatically as a Dataset object.
        pipeline.add_input(name = 'inputs')

        # Step 0: DatasetToDataFrame
        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DatasetToDataFramePrimitive.metadata.query())
        step_0.add_argument(name='inputs', argument_type = ArgumentType.CONTAINER, data_reference='inputs.0')
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

        # Step 2: ColumnParser
        step_2 = meta_pipeline.PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Step 3: imputer
        step_3 = meta_pipeline.PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
        step_3.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='replace')
        step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE, data=True)
        step_3.add_output('produce')        
        pipeline.add_step(step_3) 
        
        # Step 3: imputer
#        step_3 = meta_pipeline.PrimitiveStep(
#                primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
#        step_3.add_argument(
#                name = 'inputs',
#                argument_type=d3m_base.ArgumentType.CONTAINER,
#                data_reference='steps.2.produce')
#        step_3.add_hyperparameter(
#                name = 'use_semantic_types',
#                argument_type=d3m_base.ArgumentType.VALUE,
#                data=True
#        )
#        step_3.add_output('produce')
#        pipeline.add_step(step_3)          


        # Step 4: Extract Attributes
        step_4 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
        step_4.add_output('produce')
        step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
        pipeline.add_step(step_4)
        

        # Step 5: Extract Targets
        step_5 = meta_pipeline.PrimitiveStep(primitive_description = ExtractColumnsBySemanticTypesPrimitive.metadata.query())
        step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
        step_5.add_output('produce')
        step_5.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
        pipeline.add_step(step_5)

        #Transform attributes dataframe into an ndarray
        step_6 = meta_pipeline.PrimitiveStep(primitive_description = DataFrameToNDArrayPrimitive.metadata.query())
        step_6.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.4.produce' #inputs here are the outputs from step 3
        )
        step_6.add_output('produce')
        pipeline.add_step(step_6)

        #Run L1LowRank
        step_7 = meta_pipeline.PrimitiveStep(primitive_description = RandomizedPolyPCA.metadata.query())
        step_7.add_argument(
            name = 'inputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.6.produce' #inputs here are the outputs from step 4
        )
        step_7.add_hyperparameter(
               name = 'n_components',
               argument_type = ArgumentType.VALUE,
               data = 8
        )
        step_7.add_hyperparameter(
               name = 'degree',
               argument_type = ArgumentType.VALUE,
               data = 1
        )      
        step_7.add_output('produce')
        pipeline.add_step(step_7)
        
        # convert numpy-formatted attribute data to a dataframe
        step_8 = meta_pipeline.PrimitiveStep(primitive_description=NDArrayToDataFramePrimitive.metadata.query())
        step_8.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.7.produce'  # inputs here are the outputs from step 5
        )
        step_8.add_output('produce')
        pipeline.add_step(step_8)


        #Linear Regression on low-rank data (inputs and outputs for sklearns are both dataframes)
        step_9 = meta_pipeline.PrimitiveStep(primitive_description = d3m.primitives.classification.gradient_boosting.SKlearn.metadata.query())
        step_9.add_argument(
        	name = 'inputs',
        	argument_type = ArgumentType.CONTAINER,
        	data_reference = 'steps.8.produce'
        )
        step_9.add_argument(
            name = 'outputs',
            argument_type = ArgumentType.CONTAINER,
            data_reference = 'steps.5.produce'
        )
        step_9.add_hyperparameter(
            name = 'n_estimators',
            argument_type = ArgumentType.VALUE,
            #data = 25000
            data = 250
        )
        step_9.add_hyperparameter(
            name = 'learning_rate',
            argument_type = ArgumentType.VALUE,
            data = 0.05
        )
        step_9.add_hyperparameter(
            name = 'max_depth',
            argument_type = ArgumentType.VALUE,
            data = 2
        )                 
        step_9.add_output('produce')
        pipeline.add_step(step_9)


        #finally generate a properly-formatted output dataframe from the prediction outputs using the input dataframe as a reference
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
    random.seed(123)
    instance = randomizedpolypcaPipeline2()
    json_info = instance.get_json()
    instanceid = instance.get_id()
    instancepath = os.path.join(".", instanceid)
    with open(instancepath + ".json", 'w') as file:
        file.write(json_info)
        file.close()
