#!/bin/bash 

TensorMachinesBinaryClassificationPipeline_path="ICSI/d3m.primitives.classification.tensor_machines_binary_classification.TensorMachinesBinaryClassification/3.0.1"
TensorMachinesBinaryClassificationPipeline_id="3f6d2464-7be0-4bf4-a5ab-8f510c951e1a"
sparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline_id="02c44e21-5f5d-4ea8-b815-25ec2fdcdafd"
sparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline2_id="a7f321d7-761a-4541-a3f7-84ce24af24f6"
sparsepcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline3_id="352f740d-4a6e-4758-9bb5-66dc916ba0a8"
sparsepcaPipeline4_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline4_id="346fb1f7-6ddd-4d61-be92-e3932c983b35"
robustsparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline_id="c52d2724-86cf-486b-a885-0cfda23f785d"
randomizedpolypcaPipeline_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline_id="5a755c78-17ed-4379-bf95-f71a281dc1e6"
randomizedpolypcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline2_id="c934876b-feb7-4215-8e78-7d3f60722e3a"





####################################################
echo 'SparsePCA  -- 26_radon_seed - pipeline 1'
####################################################
#id_run="${sparsepcaPipeline_id}_run"
#path_1="$sparsepcaPipeline_path/pipelines/$sparsepcaPipeline_id.json"
#path_2="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/26_radon_seed_MIN_METADATA_problem/problemDoc.json"
#path_3="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/26_radon_seed_MIN_METADATA_dataset/datasetDoc.json"
#path_4="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
#path_5="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
#path_6="$sparsepcaPipeline_path/pipelines/$id_run.yaml"
#
##echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6
#
#`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`
#
#`echo mkdir $sparsepcaPipeline_path/pipeline_runs`
#
#`echo gzip $sparsepcaPipeline_path/pipelines/$id_run.yaml`
#
#`echo mv $sparsepcaPipeline_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline_path/pipeline_runs`


####################################################
echo 'SparsePCA  -- 534_cps_85_wages_MIN_METADATA - pipeline 2'
####################################################
id_run="${sparsepcaPipeline2_id}_run"
path_1="$sparsepcaPipeline2_path/pipelines/$sparsepcaPipeline2_id.json"
path_2="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline2_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $sparsepcaPipeline2_path/pipeline_runs`

`echo gzip $sparsepcaPipeline2_path/pipelines/$id_run.yaml`

`echo mv $sparsepcaPipeline2_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline2_path/pipeline_runs`


####################################################
echo 'SparsePCA  -- LL0_207_autoPrice_MIN_METADATA - pipeline 3'
####################################################
id_run="${sparsepcaPipeline3_id}_run"
path_1="$sparsepcaPipeline3_path/pipelines/$sparsepcaPipeline3_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline3_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $sparsepcaPipeline3_path/pipeline_runs`

`echo gzip $sparsepcaPipeline3_path/pipelines/$id_run.yaml`

`echo mv $sparsepcaPipeline3_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline3_path/pipeline_runs`


####################################################
echo 'SparsePCA  -- 196_autoMpg - pipeline 4'
####################################################
id_run="${sparsepcaPipeline4_id}_run"
path_1="$sparsepcaPipeline4_path/pipelines/$sparsepcaPipeline4_id.json"
path_2="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/196_autoMpg_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/196_autoMpg_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline4_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $sparsepcaPipeline4_path/pipeline_runs`

`echo gzip $sparsepcaPipeline4_path/pipelines/$id_run.yaml`

`echo mv $sparsepcaPipeline4_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline4_path/pipeline_runs`



####################################################
echo 'RobustSparsePCA  -- LL0_207_autoPrice_MIN_METADATA'
####################################################
id_run="${robustsparsepcaPipeline_id}_run"
path_1="$robustsparsepcaPipeline_path/pipelines/$robustsparsepcaPipeline_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$robustsparsepcaPipeline_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $robustsparsepcaPipeline_path/pipeline_runs`

`echo gzip $robustsparsepcaPipeline_path/pipelines/$id_run.yaml`

`echo mv $robustsparsepcaPipeline_path/pipelines/$id_run.yaml.gz $robustsparsepcaPipeline_path/pipeline_runs`



####################################################
echo 'RandomizedPolyPCA  -- LL0_207_autoPrice_MIN_METADATA - pipeline 1'
####################################################
id_run="${randomizedpolypcaPipeline_id}_run"
path_1="$randomizedpolypcaPipeline_path/pipelines/$randomizedpolypcaPipeline_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$randomizedpolypcaPipeline_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $randomizedpolypcaPipeline_path/pipeline_runs`

`echo gzip $randomizedpolypcaPipeline_path/pipelines/$id_run.yaml`

`echo mv $randomizedpolypcaPipeline_path/pipelines/$id_run.yaml.gz $randomizedpolypcaPipeline_path/pipeline_runs`









####################################################
echo 'TensorMachinesBinaryClassificationPipeline  -- uu4_SPECT - pipeline 2'
####################################################
id_run="${TensorMachinesBinaryClassificationPipeline_id}_run"
path_1="$TensorMachinesBinaryClassificationPipeline_path/pipelines/$TensorMachinesBinaryClassificationPipeline_id.json"
path_2="../datasets/seed_datasets_current/uu4_SPECT_MIN_METADATA/uu4_SPECT_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/uu4_SPECT_MIN_METADATA/uu4_SPECT_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/uu4_SPECT_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/uu4_SPECT_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$TensorMachinesBinaryClassificationPipeline_path/pipelines/$id_run.yaml"

echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $TensorMachinesBinaryClassificationPipeline_path/pipeline_runs`

`echo gzip $TensorMachinesBinaryClassificationPipeline_path/pipelines/$id_run.yaml`

`echo mv $TensorMachinesBinaryClassificationPipeline_path/pipelines/$id_run.yaml.gz $TensorMachinesBinaryClassificationPipeline_path/pipeline_runs`



####################################################
echo 'RandomizedPolyPCA  -- uu5_heartstatlog - pipeline 2'
####################################################
id_run="${randomizedpolypcaPipeline2_id}_run"
path_1="$randomizedpolypcaPipeline2_path/pipelines/$randomizedpolypcaPipeline2_id.json"
path_2="../datasets/seed_datasets_current/SEMI_1053_jm1_MIN_METADATA/SEMI_1053_jm1_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/SEMI_1053_jm1_MIN_METADATA/SEMI_1053_jm1_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/SEMI_1053_jm1_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/SEMI_1053_jm1_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$randomizedpolypcaPipeline2_path/pipelines/$id_run.yaml"

echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $randomizedpolypcaPipeline2_path/pipeline_runs`

`echo gzip $randomizedpolypcaPipeline2_path/pipelines/$id_run.yaml`

`echo mv $randomizedpolypcaPipeline2_path/pipelines/$id_run.yaml.gz $randomizedpolypcaPipeline2_path/pipeline_runs`
