# helper code copied from Michigan SPIDER pipeline examples, originally at
# https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/all.py
# Information on all pipelines and which primitives are used.

import json

from realML.pipelines.base import BasePipeline
from realML.pipelines.RFMPreconditionedGaussianKRR_pipeline import RFMPreconditionedGaussianKRRPipeline

PIPELINES_BY_PRIMITIVE = {
    'd3m.primitives.realML.RFMPreconditionedGaussianKRR': [
        RFMPreconditionedGaussianKRRPipeline,
    ],
}

def get_primitives():
    return PIPELINES_BY_PRIMITIVE.keys()

def get_pipelines(primitive = None):
    if (primitive is not None):
        if (primitive not in PIPELINES_BY_PRIMITIVE):
            return []
        return PIPELINES_BY_PRIMITIVE[primitive]

    pipelines = set()
    for primitive_pipelines in PIPELINES_BY_PRIMITIVE.values():
        pipelines = pipelines | set(primitive_pipelines)
    return pipelines

if __name__ == '__main__':
    print(json.dumps(PIPELINES_BY_PRIMITIVE))


