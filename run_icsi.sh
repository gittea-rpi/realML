#!/bin/bash 


RFMPreconditionedGaussianKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_id="8aa30a22-7c52-4a17-bb51-d79f4bbf7972"
RFMPreconditionedGaussianKRRPipeline_196_autoMpg_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_196_autoMpg_id="1ad0a50a-c6db-4c1e-b0d4-d6c3e34df171"
RFMPreconditionedPolynomialKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR/3.0.1"
RFMPreconditionedPolynomialKRRPipeline_id="76cc7654-a137-4775-b8e3-5bfd4cdbf8a7"
TensorMachinesRegularizedLeastSquaresPipeline_path="ICSI/d3m.primitives.regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares/3.0.1"
TensorMachinesRegularizedLeastSquaresPipeline_id="802e744c-22bb-4113-8028-61c41c10a0b0"
sparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline2_id="a49a604a-a82b-4f40-9ec8-ace9fe4dbcc4"
sparsepcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline3_id="9d2da2f6-13ef-4b64-b820-9d4d40fc8622"
sparsepcaPipeline4_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline4_id="1aea62dc-5dd6-4229-9d9f-59afc68694a6"
sparsepcaPipeline5_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline5_id="48c8374f-c121-4a3a-85de-faaee57726aa"
robustsparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline_id="bcc3ab65-2094-4a8c-ae04-cc8a250d8377"
robustsparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline2_id="11feb395-8229-4592-bfd2-b00ce8c36f64"
randomizedpolypcaPipeline_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline_id="5234a868-ca6e-49ef-b0ea-e0d30174cb46"
randomizedpolypcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline3_id="ddf5ba15-90d4-40e4-8f1a-8809a77189b0"




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
echo 'SparsePCA  -- 56_sunspots_MIN_METADATA - pipeline 5'
####################################################
id_run="${sparsepcaPipeline5_id}_run"
path_1="$sparsepcaPipeline5_path/pipelines/$sparsepcaPipeline5_id.json"
path_2="../datasets/seed_datasets_current/56_sunspots_MIN_METADATA/56_sunspots_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/56_sunspots_MIN_METADATA/56_sunspots_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/56_sunspots_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/56_sunspots_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$sparsepcaPipeline5_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $sparsepcaPipeline5_path/pipeline_runs`

`echo gzip $sparsepcaPipeline5_path/pipelines/$id_run.yaml`

`echo mv $sparsepcaPipeline5_path/pipelines/$id_run.yaml.gz $sparsepcaPipeline5_path/pipeline_runs`


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



