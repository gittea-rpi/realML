#!/bin/bash 



id_sparsePCA_pip3="8758bb1a-c4b4-42aa-8bff-8a06fff54980" 

####################################################
# SparsePCA  -- LL0_207_autoPrice_MIN_METADATA
####################################################

python3 -m d3m runtime fit-score -p ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3.json -r ../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json -i ../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json -t ../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json -a ../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json -O ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3_run.yaml

mkdir ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipeline_runs

gzip ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3_run.yaml

mv ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipelines/$id_sparsePCA_pip3_run.yaml.gz ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.5/pipeline_runs

echo "Run SparsePCA  for LL0_207_autoPrice_MIN_METADATA"