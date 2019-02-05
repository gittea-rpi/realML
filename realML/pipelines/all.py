# helper code copied from Michigan SPIDER pipeline examples, originally at
# https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/all.py
# Information on all pipelines and which primitives are used.

import json

from realML.pipelines.base import BasePipeline
from realML.pipelines.RFMPreconditionedGaussianKRR_pipeline import RFMPreconditionedGaussianKRRPipeline
from realML.pipelines.RFMPreconditionedPolynomialKRR_pipeline import RFMPreconditionedPolynomialKRRPipeline
from realML.pipelines.TensorMachinesRegularizedLeastSquares_pipeline import TensorMachinesRegularizedLeastSquaresPipeline
from realML.pipelines.FastLAD_pipeline import FastLADPipeline

PIPELINES_BY_PRIMITIVE = {
    'd3m.primitives.realML.RFMPreconditionedGaussianKRR': [
        RFMPreconditionedGaussianKRRPipeline,
    ],
    'd3m.primitives.realML.RFMPreconditionedPolynomialKRR': [
        RFMPreconditionedPolynomialKRRPipeline,
    ],
    'd3m.primitives.realML.TensorMachinesRegularizedLeastSquares': [
        TensorMachinesRegularizedLeastSquaresPipeline,
    ],
    'd3m.primitives.realML.FastLAD': [
        FastLADPipeline,
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


