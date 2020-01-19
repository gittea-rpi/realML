#!/bin/bash 

RFMPreconditionedGaussianKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_id="538d8de1-3321-4f07-b97f-0be03b57b167"
RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_196_autoMpg_id="95479984-d1b0-429d-9d76-58d8a35519b7"
RFMPreconditionedPolynomialKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR/3.0.1"
RFMPreconditionedPolynomialKRRPipeline_id="faa88202-4ba7-462a-a5c4-bdc6a1385a5d"
TensorMachinesRegularizedLeastSquaresPipeline_path="ICSI/d3m.primitives.regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares/3.0.1"
TensorMachinesRegularizedLeastSquaresPipeline_id="685c6c1c-0cab-41b4-8e2a-1acccd1d0a3f"
sparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline2_id="9c8c7f11-01f6-4158-b65f-105987f8bf31"
sparsepcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline3_id="8291f380-551b-4e57-967a-006ce2624acc"
sparsepcaPipeline4_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline4_id="687fd5a6-886d-466c-a65a-e495b850b071"
sparsepcaPipeline5_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline5_id="fcee0795-dadd-4e90-a1e0-a6d95b2fd1ca"
robustsparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline_id="0626d1b6-858d-4070-93ad-33b9682a0ce5"
robustsparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline2_id="cc057e01-7856-41d9-983a-b6062324d2c5"
randomizedpolypcaPipeline_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline_id="08906a28-5b83-43b4-8cff-144e4c91b1e7"
randomizedpolypcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline3_id="195277ad-3dcd-478a-9aa6-19d791517ce9"




####################################################
echo 'SparsePCA  -- uu3_world_development_indicators_MIN_METADATA - pipeline 5'
####################################################
id_run="${sparsepcaPipeline5_id}_run"
path_1="$sparsepcaPipeline5_path/pipelines/$sparsepcaPipeline5_id.json"
path_2="../datasets/seed_datasets_current/uu3_world_development_indicators_MIN_METADATA/uu3_world_development_indicators_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/uu3_world_development_indicators_MIN_METADATA/uu3_world_development_indicators_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/uu3_world_development_indicators_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/uu3_world_development_indicators_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline5_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $sparsepcaPipeline5_path/pipeline_runs`

`echo gzip $sparsepcaPipeline5_path/pipelines/$id_run.yaml`

`echo mv $sparsepcaPipeline5_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline5_path/pipeline_runs`



####################################################
echo 'TensorMachinesRegularizedLeastSquares  -- LL0_207_autoPrice_MIN_METADATA'
####################################################
id_run="${TensorMachinesRegularizedLeastSquaresPipeline_id}_run"
path_1="$TensorMachinesRegularizedLeastSquaresPipeline_path/pipelines/$TensorMachinesRegularizedLeastSquaresPipeline_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$TensorMachinesRegularizedLeastSquaresPipeline_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $TensorMachinesRegularizedLeastSquaresPipeline_path/pipeline_runs`

`echo gzip $TensorMachinesRegularizedLeastSquaresPipeline_path/pipelines/$id_run.yaml`

`echo mv $TensorMachinesRegularizedLeastSquaresPipeline_path/pipelines/$id_run.yaml.gz $TensorMachinesRegularizedLeastSquaresPipeline_path/pipeline_runs`


####################################################
echo 'RFMPreconditionedGaussianKRRP  -- LL0_207_autoPrice_MIN_METADATA'
####################################################
id_run="${RFMPreconditionedGaussianKRRPipeline_id}_run"
path_1="$RFMPreconditionedGaussianKRRPipeline_path/pipelines/$RFMPreconditionedGaussianKRRPipeline_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$RFMPreconditionedGaussianKRRPipeline_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $RFMPreconditionedGaussianKRRPipeline_path/pipeline_runs`

`echo gzip $RFMPreconditionedGaussianKRRPipeline_path/pipelines/$id_run.yaml`

`echo mv $RFMPreconditionedGaussianKRRPipeline_path/pipelines/$id_run.yaml.gz $RFMPreconditionedGaussianKRRPipeline_path/pipeline_runs`


####################################################
echo 'RFMPreconditionedGaussianKRRP  -- LL0_207_autoPrice_MIN_METADATA'
####################################################
id_run="${RFMPreconditionedGaussianKRRPipeline_196_autoMpg_id}_run"
path_1="$RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipelines/$RFMPreconditionedGaussianKRRPipeline_196_autoMpg_id.json"
path_2="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/196_autoMpg_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/196_autoMpg_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/196_autoMpg_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipeline_runs`

`echo gzip $RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipelines/$id_run.yaml`

`echo mv $RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipelines/$id_run.yaml.gz $RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path/pipeline_runs`



####################################################
echo 'RFMPreconditionedPolynomialKRR  -- LL0_207_autoPrice_MIN_METADATA'
####################################################
id_run="${RFMPreconditionedPolynomialKRRPipeline_id}_run"
path_1="$RFMPreconditionedPolynomialKRRPipeline_path/pipelines/$RFMPreconditionedPolynomialKRRPipeline_id.json"
path_2="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/LL0_207_autoPrice_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/LL0_207_autoPrice_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$RFMPreconditionedPolynomialKRRPipeline_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

`echo mkdir $RFMPreconditionedPolynomialKRRPipeline_path/pipeline_runs`

`echo gzip $RFMPreconditionedPolynomialKRRPipeline_path/pipelines/$id_run.yaml`

`echo mv $RFMPreconditionedPolynomialKRRPipeline_path/pipelines/$id_run.yaml.gz $RFMPreconditionedPolynomialKRRPipeline_path/pipeline_runs`



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

#`echo mkdir $sparsepcaPipeline3_path/pipeline_runs`

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

#`echo mkdir $sparsepcaPipeline4_path/pipeline_runs`

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
echo 'RobustSparsePCA  -- 534_cps_85_wages - pipeline2'
####################################################
id_run="${robustsparsepcaPipeline2_id}_run"
path_1="$robustsparsepcaPipeline2_path/pipelines/$robustsparsepcaPipeline2_id.json"
path_2="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$robustsparsepcaPipeline2_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $robustsparsepcaPipeline2_path/pipeline_runs`

`echo gzip $robustsparsepcaPipeline2_path/pipelines/$id_run.yaml`

`echo mv $robustsparsepcaPipeline2_path/pipelines/$id_run.yaml.gz $robustsparsepcaPipeline2_path/pipeline_runs`



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
echo 'RandomizedPolyPCA  -- 534_cps_85_wages - pipeline 3'
####################################################
id_run="${randomizedpolypcaPipeline3_id}_run"
path_1="$randomizedpolypcaPipeline3_path/pipelines/$randomizedpolypcaPipeline3_id.json"
path_2="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/534_cps_85_wages_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/534_cps_85_wages_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$randomizedpolypcaPipeline3_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $randomizedpolypcaPipeline3_path/pipeline_runs`

`echo gzip $randomizedpolypcaPipeline3_path/pipelines/$id_run.yaml`

`echo mv $randomizedpolypcaPipeline3_path/pipelines/$id_run.yaml.gz $randomizedpolypcaPipeline3_path/pipeline_runs`



