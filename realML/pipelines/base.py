# helper class copied from Michigan SPIDER team's pipeline 
# originally at https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/base.py
import abc
import json

class BasePipeline(object):
    def __init__(self):
        self._pipeline = self._gen_pipeline()

    @abc.abstractmethod
    def _gen_pipeline(self):
        '''
        Create a D3M pipeline for this class.
        '''
        pass

    @abc.abstractmethod
    def assert_result(self, tester, results, dataset):
        '''
        Make sure that the results from an invocation of this pipeline are valid.
        '''
        pass

    def get_id(self):
        return self._pipeline.id

    def genmeta(self, dataset):
        meta_info = {
            'problem': dataset + "_problem",
            'full_inputs': [ dataset + "_dataset" ],
            'train_inputs': [ dataset + "_dataset_TRAIN" ],
            'test_inputs': [ dataset + "_dataset_TEST" ],
            'score_inputs': [ dataset + "_dataset_SCORE" ]}
        return meta_info

    def get_json(self):
        # Make it pretty.
        return json.dumps(json.loads(self._pipeline.to_json()), indent = 4)

