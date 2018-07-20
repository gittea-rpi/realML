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

    def get_json(self):
        # Make it pretty.
        return json.dumps(json.loads(self._pipeline.to_json()), indent = 4)

