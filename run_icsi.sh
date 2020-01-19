#!/bin/bash 


RFMPreconditionedGaussianKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_id="3c2e0f55-dd32-4c6c-80a5-a36e3af8f9e0"
RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR/3.0.1"
RFMPreconditionedGaussianKRRPipeline_26_radon_seed_id="69a89cf6-ba2c-41db-aab9-64208e74680f"
RFMPreconditionedPolynomialKRRPipeline_path="ICSI/d3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR/3.0.1"
RFMPreconditionedPolynomialKRRPipeline_id="c99abbf6-710e-4027-9fa1-6f7e328e81c7"
sparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline2_id="09ec5ff1-a907-4704-a00d-cbe77b3bbe06"
sparsepcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline3_id="4f5599c3-2ebb-4ca2-b673-34d481f50d7b"
sparsepcaPipeline4_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.SparsePCA/3.0.1"
sparsepcaPipeline4_id="46a65cb6-be60-473e-a853-7e7b23dc6c7e"
robustsparsepcaPipeline_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline_id="1ff4b6ab-f1d4-482c-b5bd-4f04e897083d"
robustsparsepcaPipeline2_path="ICSI/d3m.primitives.feature_extraction.sparse_pca.RobustSparsePCA/3.0.1"
robustsparsepcaPipeline2_id="8cee6111-6d01-4f2d-a888-ea11d80c7ded"
randomizedpolypcaPipeline_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline_id="451ea77d-af2b-4b63-98be-bdf2cee78416"
randomizedpolypcaPipeline3_path="ICSI/d3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA/3.0.1"
randomizedpolypcaPipeline3_id="9ffff3b3-e66b-47ca-be45-a984bd13fe35"









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
echo 'RFMPreconditionedGaussianKRRP  -- 26_radon_seed'
####################################################
id_run="${RFMPreconditionedGaussianKRRPipeline_26_radon_seed_id}_run"
path_1="$RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipelines/$RFMPreconditionedGaussianKRRPipeline_26_radon_seed_id.json"
path_2="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/26_radon_seed_MIN_METADATA_problem/problemDoc.json"
path_3="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/26_radon_seed_MIN_METADATA_dataset/datasetDoc.json"
path_4="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/TEST/dataset_TEST/datasetDoc.json"
path_5="../datasets/seed_datasets_current/26_radon_seed_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json"
path_6="$RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipelines/$id_run.yaml"

#echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6

`echo python3 -m d3m runtime fit-score -p $path_1 -r $path_2 -i $path_3 -t $path_4 -a $path_5 -O $path_6`

#`echo mkdir $RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipeline_runs`

`echo gzip $RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipelines/$id_run.yaml`

`echo mv $RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipelines/$id_run.yaml.gz $RFMPreconditionedGaussianKRRPipeline_26_radon_seed_path/pipeline_runs`



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



