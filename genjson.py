#!/usr/bin/env python3
#convenience script for generating the entire primitive JSON file structure 
#copied from the Michigan SPIDER repo, originally at https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/genjson.py

import subprocess
import os, os.path
import shutil
import sys
import pkg_resources
import json
from realML.pipelines import *

#clear out any existing directory
if os.path.isdir("ICSI"):
    shutil.rmtree("ICSI")

os.makedirs("ICSI")

version = pkg_resources.get_distribution("realML").version

primitives = {
    'd3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR'  : [RFMPreconditionedGaussianKRRPipeline],
    'd3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR' : [RFMPreconditionedPolynomialKRRPipeline],
    'd3m.primitives.regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares': [TensorMachinesRegularizedLeastSquaresPipeline],
    'd3m.primitives.classification.tensor_machines_binary_classification.TensorMachinesBinaryClassification' : None,
    'd3m.primitives.regression.fast_lad.FastLAD' : [FastLADPipeline],
    'd3m.primitives.feature_extraction.l1_low_rank.L1LowRank' : None
}

for prim in primitives.keys():
    path = os.path.join("ICSI", prim)
    os.makedirs(path)
    path = os.path.join(path, version)
    os.makedirs(path)

    com = "python3 -m d3m.index describe -i 4 " + prim + " > " + os.path.join(path, "primitive.json")
    print('Running command: %s' % str(com))
    subprocess.check_call(com, shell=True)

    #now make pipelines
    #if primitives[prim] is not None:
    #    plpath = os.path.join(path, 'pipelines')
    #    os.makedirs(plpath)
    #    
    #    pls = primitives[prim]
    #    for pl in pls:
    #        instance = pl()
    #        json_info = instance.get_json()
    #        instanceid = instance.get_id()

    #        instancepath = os.path.join(plpath, instanceid)
    #        with open(instancepath + ".json", 'w') as file:
    #            file.write(json_info)
    #            file.close()
    #        
    #        meta = instance.meta_info
    #        with open(instancepath + ".meta", 'w') as file:
    #            json.dump(meta, file, indent = 4)
    #            file.close()
    
    


