# helper class taken from Michigan SPIDER team's pipeline examples, originally at 
# https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/datasets.py

import json

# All datasets with their sizes.
DATASETS = {
    '1491_one_hundred_plants_margin': 319,
    '1491_one_hundred_plants_margin_clust': 319,
    '1567_poker_hand': 205001,
    '185_baseball': 267,
    '196_autoMpg': 100,
    '22_handgeometry': 38,
    '26_radon_seed': 183,
    '27_wordLevels': 1400,
    '299_libras_move': 100,
    '30_personae': 29,
    '313_spectrometer': 103,
    '31_urbansound': 339,
    '32_wikiqa': 5852,
    '38_sick': 754,
    '4550_MiceProtein': 215,
    '49_facebook': 1208,
    '534_cps_85_wages': 106,
    '56_sunspots': 29,
    '57_hypothyroid': 754,
    '59_umls': 3409,
    '60_jester': 880720,
    '66_chlorineConcentration': 3840,
    '6_70_com_amazon': 62462,
    '6_86_com_DBLP': 107228,
    'DS01876': 1759,
    'LL1_3476_HMDB_actio_recognition':70,
    'LL0_207_autoPrice': 158,
    'LL1_net_nomination_seed': 80,
    'LL1_penn_fudan_pedestrian': 134,
    'uu1_datasmash': 45,
    'uu2_gp_hyperparameter_estimation': 100,
    'uu3_world_development_indicators': 2461,
    'uu4_SPECT': 81,
}

TRAIN_IDS = {
    '1491_one_hundred_plants_margin': '1491_one_hundred_plants_dataset_TRAIN',
    '1491_one_hundred_plants_margin_clust': '1491_one_hundred_plants_margin_dataset_TRAIN',
    '1567_poker_hand': '1567_poker_dataset_TRAIN',
    '185_baseball': '185_bl_dataset_TRAIN',
    '196_autoMpg': '196_ag_dataset_TRAIN',
    '22_handgeometry': '22_hy_dataset_TRAIN',
    '26_radon_seed': '26_radon_dataset_TRAIN',
    '27_wordLevels': '27_ws_dataset_TRAIN',
    '299_libras_move': '299_libras_dataset_TRAIN',
    '30_personae': '30_pe_dataset_TRAIN',
    '313_spectrometer': '313_sr_dataset_TRAIN',
    '31_urbansound': '31_ud_dataset_TRAIN',
    '32_wikiqa': '32_wa_dataset_TRAIN',
    '38_sick': '38_sk_dataset_TRAIN',
    '4550_MiceProtein': '4550_Mn_dataset_TRAIN',
    '49_facebook': '49_fk_dataset_TRAIN',
    '534_cps_85_wages': '534_cps_85_dataset_TRAIN',
    '56_sunspots': '56_ss_dataset_TRAIN',
    '57_hypothyroid': '57_hd_dataset_TRAIN',
    '59_umls': '59_us_dataset_TRAIN',
    '60_jester': '60_jr_dataset_TRAIN',
    '66_chlorineConcentration': '66_cn_dataset_TRAIN',
    '6_70_com_amazon': '6_70_com_dataset_TRAIN',
    '6_86_com_DBLP': '6_86_com_dataset_TRAIN',
    'DS01876': 'DS01876_dataset_TRAIN',
    'LL0_207_autoPrice': 'LL0_207_dataset_TRAIN',
    'LL1_net_nomination_seed': 'LL1_net_nomination_dataset_TRAIN',
    'LL1_penn_fudan_pedestrian': 'penn_fudan_pedestrian_dataset',
    'LL1_3476_HMDB_actio_recognition': 'LL1_3476_HMDB_actio_dataset_TRAIN',
    'uu1_datasmash': 'uu1_dh_dataset_TRAIN',
    'uu2_gp_hyperparameter_estimation': 'uu2_gp_hyperparameter_dataset_TRAIN',
    'uu3_world_development_indicators': 'uu3_world_development_dataset_TRAIN',
    'uu4_SPECT': 'uu4_ST_dataset_TRAIN',
}

TEST_IDS = {
    '1491_one_hundred_plants_margin': '1491_one_hundred_plants_dataset_TEST',
    '1491_one_hundred_plants_margin_clust': '1491_one_hundred_plants_margin_dataset_TEST',
    '1567_poker_hand': '1567_poker_dataset_TEST',
    '185_baseball': '185_bl_dataset_TEST',
    '196_autoMpg': '196_ag_dataset_TEST',
    '22_handgeometry': '22_hy_dataset_TEST',
    '26_radon_seed': '26_radon_dataset_TEST',
    '27_wordLevels': '27_ws_dataset_TEST',
    '299_libras_move': '299_libras_dataset_TEST',
    '30_personae': '30_pe_dataset_TEST',
    '313_spectrometer': '313_sr_dataset_TEST',
    '31_urbansound': '31_ud_dataset_TEST',
    '32_wikiqa': '32_wa_dataset_TEST',
    '38_sick': '38_sk_dataset_TEST',
    '4550_MiceProtein': '4550_Mn_dataset_TEST',
    '49_facebook': '49_fk_dataset_TEST',
    '534_cps_85_wages': '534_cps_85_dataset_TEST',
    '56_sunspots': '56_ss_dataset_TEST',
    '57_hypothyroid': '57_hd_dataset_TEST',
    '59_umls': '59_us_dataset_TEST',
    '60_jester': '60_jr_dataset_TEST',
    '66_chlorineConcentration': '66_cn_dataset_TEST',
    '6_70_com_amazon': '6_70_com_dataset_TEST',
    '6_86_com_DBLP': '6_86_com_dataset_TEST',
    'DS01876': 'DS01876_dataset_TEST',
    'LL0_207_autoPrice': 'LL0_207_dataset_TEST',
    'LL1_net_nomination_seed': 'LL1_net_nomination_dataset_TEST',
    'LL1_penn_fudan_pedestrian': 'penn_fudan_pedestrian_dataset',
    'LL1_3476_HMDB_actio_recognition': 'LL1_3476_HMDB_actio_dataset_TEST',
    'uu1_datasmash': 'uu1_dh_dataset_TEST',
    'uu2_gp_hyperparameter_estimation': 'uu2_gp_hyperparameter_dataset_TEST',
    'uu3_world_development_indicators': 'uu3_world_development_dataset_TEST',
    'uu4_SPECT': 'uu4_ST_dataset_TEST',
}

PROBLEM_IDS = {
    '1491_one_hundred_plants_margin': '1491_one_hundred_plants_margin_problem',
    '1491_one_hundred_plants_margin_clust': '1491_one_hundred_plants_margin_clust_problem',
    '1567_poker_hand': '1567_poker_hand_problem',
    '185_baseball': '185_baseball_problem',
    '196_autoMpg': '196_autoMpg_problem',
    '22_handgeometry': '22_handgeometry_problem',
    '26_radon_seed': '26_radon_problem',
    '27_wordLevels': '27_wordLevels_problem',
    '299_libras_move': '299_libras_move_problem',
    '30_personae': '30_personae_problem',
    '313_spectrometer': '313_spectrometer_problem',
    '31_urbansound': '31_urbansound_problem',
    '32_wikiqa': '32_wikiqa_problem',
    '38_sick': '38_sick_problem',
    '4550_MiceProtein': '4550_MiceProtein_problem',
    '49_facebook': '49_facebook_problem',
    '534_cps_85_wages': '534_cps_85_wages_problem',
    '56_sunspots': '56_sunspots_problem',
    '57_hypothyroid': '57_hypothyroid_problem',
    '59_umls': '59_umls_problem',
    '60_jester': '60_jester_problem',
    '66_chlorineConcentration': '66_chlorineConcentration_problem',
    '6_70_com_amazon': '6_70_com_amazon_problem',
    '6_86_com_DBLP': '6_86_com_DBLP_problem',
    'DS01876': 'DS01876_problem',
    'LL0_207_autoPrice': 'LL0_207_autoPrice_problem',
    'LL1_net_nomination_seed': 'LL1_net_nomination_seed_problem',
    'LL1_penn_fudan_pedestrian': 'penn_fudan_pedestrian_problem',
    'LL1_3476_HMDB_actio_recognition': 'LL1_3476_HMDB_actio_recognition_problem',
    'uu1_datasmash': 'uu1_datasmash_problem',
    'uu2_gp_hyperparameter_estimation': 'uu2_gp_hyperparameter_estimation_problem',
    'uu3_world_development_indicators': 'uu3_world_development_indicators_problem',
    'uu4_SPECT': 'uu4_SPECT_problem',
}

def get_datasets():
    return DATASETS

def get_dataset_ids():
    return 

def get_count(dataset):
    return DATASETS[dataset]

def get_dataset_names():
    return DATASETS.keys()

def get_train_id(dataset_id):
    return TRAIN_IDS[dataset_id]

def get_test_id(dataset_id):
    return TEST_IDS[dataset_id]

def get_problem_id(dataset_id):
    return PROBLEM_IDS[dataset_id]

if __name__ == '__main__':
    print(json.dumps(get_datasets(), indent = 4))

