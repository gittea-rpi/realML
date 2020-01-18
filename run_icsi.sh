#!/bin/bash 

version="2.8.6"

sparsepcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/2.8.6/pipelines"
sparsepcaPipeline3_id="23a9f478-7141-4559-96ed-2780da618494"
robustsparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/2.8.6/pipelines"
robustsparsepcaPipeline_id="a5f04134-6ea6-4040-9d15-1281e0c7c196"
randomizedpolypcaPipeline_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/2.8.6/pipelines"
randomizedpolypcaPipeline_id="3e612a40-5122-4792-ba8f-ec9de38efcd8"

####################################################
# SparsePCA  -- LL0_207_autoPrice_MIN_METADATA
####################################################
id_run="${sparsepcaPipeline3_id}_run"
path_1="$sparsepcaPipeline3_path/$sparsepcaPipeline3_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline3_path/$id_run.yaml"

echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6
`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/$version/pipeline_runs`

`echo gzip ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/$version/pipelines/$id_run.yaml`

`echo mv ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/$version/pipelines/$id_run.yaml.gz ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/$version/pipeline_runs`

