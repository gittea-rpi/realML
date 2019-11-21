#!/usr/bin/env python3
#
#convenience script for generating the entire primitive JSON file structure 
#modified from the Michigan SPIDER repo, originally at https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/genjson.py

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
    'd3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR'  : [
        RFMPreconditionedGaussianKRRPipeline, 
    #    RFMPreconditionedGaussianKRRPipeline_196_autoMpg,
    #    RFMPreconditionedGaussianKRRPipeline_26_radon_seed,
     ],
    'd3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR' : [
        RFMPreconditionedPolynomialKRRPipeline, 
    #    RFMPreconditionedPolynomialKRRPipeline_196_autoMpg,
    #    RFMPreconditionedPolynomialKRRPipeline_26_radon_seed
     ],
    'd3m.primitives.regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares': [
        TensorMachinesRegularizedLeastSquaresPipeline,
    #    TensorMachinesRegularizedLeastSquaresPipeline_196_autoMpg,
    #    TensorMachinesRegularizedLeastSquaresPipeline_26_radon_seed,
     ],
    'd3m.primitives.classification.tensor_machines_binary_classification.TensorMachinesBinaryClassification' : [TensorMachinesBinaryClassificationPipeline],
    'd3m.primitives.regression.fast_lad.FastLAD' : [FastLADPipeline],
    'd3m.primitives.feature_extraction.l1_low_rank.L1LowRank' : [L1LowRankPipeline],
    #'d3m.primitives.feature_extraction.sparse_pca.SparsePCA' : [
#	sparsepcaPipeline,
#	sparsepcaPipeline2,
#	sparsepcaPipeline3,
#	sparsepcaPipeline4
#	],
    'd3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA' : [
	robustsparsepcaPipeline
	],
    'd3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA' : [
#	randomizedpolypcaPipeline,
	randomizedpolypcaPipeline2
	],
}

for prim in primitives.keys():
    if primitives[prim] is not None: # only submit primitives with pipelines
        path = os.path.join("ICSI", prim)
        os.makedirs(path)
        path = os.path.join(path, version)
        os.makedirs(path)

        com = "python3 -m d3m index describe -i 4 " + prim + " > " + os.path.join(path, "primitive.json")
        print('Running command: %s' % str(com))
        subprocess.check_call(com, shell=True)

        plpath = os.path.join(path, 'pipelines')
        os.makedirs(plpath)
        
        pipelines = primitives[prim]
        for pl in pipelines:
            instance = pl()
            json_info = instance.get_json()
            instanceid = instance.get_id()

            instancepath = os.path.join(plpath, instanceid)
            with open(instancepath + ".json", 'w') as file:
                file.write(json_info)
                file.close()
            
            meta = instance.meta_info
            with open(instancepath + ".meta", 'w') as file:
                json.dump(meta, file, indent = 4)
                file.close()
            
            print('For pipeline name: ' + pl.__name__)
            print('Corresponding test command: python3 -m d3m runtime -d ../datasets/ fit-score -m ' + instancepath + '.meta -p ' + instancepath + '.json')
            print(' ')
    
    


