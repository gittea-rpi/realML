#!/bin/bash 



id_sparsePCA_pip3="8758bb1a-c4b4-42aa-8bff-8a06fff54980" 


####################################################
# SparsePCA  -- LL0_207_autoPrice_MIN_METADATA
####################################################

path_1="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -O ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3_run.yaml"

run='python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5'
`echo $run`



