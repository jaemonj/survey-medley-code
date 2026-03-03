#!/usr/bin/env python3

import glob
import re
from pathlib import Path

import nibabel as nf
import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel

from survey_medley_code.analysis_provenance import log_provenance
from survey_medley_code.config_loader import load_config


def get_sub_event_file(sub_id, events_files):
    for file in events_files:
        if sub_id in file:
            return file


# Create dictionairy of questionnaires with their corresponding question numbers
def define_questionnaires():
    grit_questions = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08']
    brief_questions = [
        'Q09',
        'Q10',
        'Q11',
        'Q12',
        'Q13',
        'Q14',
        'Q15',
        'Q16',
        'Q17',
        'Q18',
        'Q19',
        'Q20',
        'Q21',
    ]
    future_time_questions = [
        'Q22',
        'Q23',
        'Q24',
        'Q25',
        'Q26',
        'Q27',
        'Q28',
        'Q29',
        'Q30',
        'Q31',
    ]
    upps_questions = ['Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37']
    impulsive_venture_questions = ['Q38', 'Q39', 'Q40']
    questionnaires = {
        'grit': grit_questions,
        'brief': brief_questions,
        'future_time': future_time_questions,
        'upps': upps_questions,
        'impulsive_venture': impulsive_venture_questions,
    }
    return questionnaires


# creates data frame with columns: questionnaire_name, question_id (e.g., Q01), behavior, chr_count, bold_file
def get_behavior_bold_data(
    sub_id, events_files, questionnaires, question_output_path, activation_maps
):
    sub_event_file = get_sub_event_file(sub_id, events_files)
    event_df = pd.read_csv(sub_event_file, sep='\t')

    questionnaire_name_list = []
    question_id_list = []
    behavior_list = []
    chr_count_list = []
    bold_file_list = []
    for questionnaire in questionnaires:
        for question in questionnaires[questionnaire]:
            good_sub_file = Path(
                f'{question_output_path}/outlier_assessment/subjects_outlier_percent_lt_8_contrast_{question}.txt'
            )
            good_sub_list = good_sub_file.read_text().splitlines()
            response = event_df.loc[
                event_df['trial_type'] == question, 'coded_response'
            ]
            behavior = response.values[0]
            question_bold_file = f'{question_output_path}/{sub_id}/contrast_{question}_effect_size_sub_{sub_id}.nii.gz'

            if (
                (question_bold_file in activation_maps)
                and (not pd.isna(behavior))
                and ('s' + sub_id in good_sub_list)
            ):
                questionnaire_name_list.append(questionnaire)
                question_id_list.append(question)
                behavior_list.append(behavior)

                text = event_df.loc[event_df['trial_type'] == question, 'item_text']
                chr_count = len(text.values[0])
                chr_count_list.append(chr_count)

                bold_file_list.append(question_bold_file)
    behavior_bold_df = pd.DataFrame(
        {
            'questionnaire_name': questionnaire_name_list,
            'question_id': question_id_list,
            'behavior': behavior_list,
            'chr_count': chr_count_list,
            'bold_file': bold_file_list,
        }
    )
    return behavior_bold_df


"""
Input: pandas data frame from get_behavior_bold_data, min_obs (lower bound for the number of items required per questionnaire, default=5), 
min_unique (minimum number of unique behavioral values, default=3), min_obs_per_level_when_binary 
(minimum number of observations per level of behavior when there are only 2 unique behavior values)

Output: subsetted pandas data frame with the omitted questionnaires removed
"""


def apply_inclusion_criteria(
    df, questionnaires, min_obs=5, min_unique=3, min_obs_per_level_when_binary=None
):
    for questionnaire in questionnaires:
        print('processing', questionnaire)
        omit = False
        filtered_df = df[df['questionnaire_name'] == questionnaire]
        num_obs = filtered_df.shape[0]
        # print("observations:", num_obs)
        if num_obs < min_obs:
            omit = True
            # print(num_obs, "observations <", min_obs, "minimum")
        num_unique = filtered_df['behavior'].nunique()
        if num_unique < min_unique:
            omit = True
            # print(num_unique, "unique obs <", min_unique, "minimum")
        if min_obs_per_level_when_binary != None and num_unique == 2:
            num_obs_per_level_series = filtered_df['behavior'].value_counts()
            lowest_obs_per_level = num_obs_per_level_series.min()
            if lowest_obs_per_level < min_obs_per_level_when_binary:
                omit = True
                # print(lowest_obs_per_level, "obs per level when binary <", min_obs_per_level_when_binary, "minimum")
        if omit:
            df = df[df['questionnaire_name'] != questionnaire]
            # print("omitting", questionnaire)
    new_df = df.reset_index(drop=True)
    return new_df


"""
Input: output from apply_inclusion_criteria
Output: design_matrix (pandas df), bold file list, contrast names (list of strings)
"""


def build_design_matrix_contrasts(df):
    design_matrix = pd.DataFrame()
    questionnaires = df.questionnaire_name.unique()
    contrast_names = []
    # columns are dummy regressors for each questionnaire and questionnaire-specific behavioral regressors
    for questionnaire in questionnaires:
        design_matrix[questionnaire] = np.where(
            df['questionnaire_name'] == questionnaire, 1, 0
        )
        design_matrix[f'{questionnaire}_responses'] = np.where(
            df['questionnaire_name'] == questionnaire, df['behavior'], 0
        )
        contrast_names.append(f'{questionnaire}_responses')
    # Add character count column to adjust for question length
    design_matrix['chr_count'] = df['chr_count']

    bold_file_list = df['bold_file']

    return design_matrix, bold_file_list, contrast_names


# generate a contrast map for each questionnaire in the given design matrix
def run_within_subject_questionnaire_glm(
    sub_id,
    events_files,
    questionnaires,
    question_output_path,
    activation_maps,
    min_obs=5,
    min_unique=3,
    min_obs_per_level_when_binary=None,
    chr_count_adjustment=True,
    cfg=load_config(),
):
    behavior_bold_df = get_behavior_bold_data(
        sub_id, events_files, questionnaires, question_output_path, activation_maps
    )
    new_df = apply_inclusion_criteria(
        behavior_bold_df,
        questionnaires,
        min_obs,
        min_unique,
        min_obs_per_level_when_binary,
    )
    design_matrix, sub_bold_files, contrast_names = build_design_matrix_contrasts(
        new_df
    )

    assert design_matrix.shape[0] == len(sub_bold_files)
    assert not design_matrix.isna().any().any()
    # return if there are no contrasts to run
    if not contrast_names:
        return

    if not chr_count_adjustment:
        design_matrix = design_matrix.drop(columns=['chr_count'])

    model = SecondLevelModel(n_jobs=2)
    model.fit(sub_bold_files, design_matrix=design_matrix)
    for contrast in contrast_names:
        contrast_map = model.compute_contrast(
            second_level_contrast=contrast, output_type='effect_size'
        )
        if not chr_count_adjustment:
            contrast_map_dir = (
                cfg.output_root
                / f'within_subject_brain_behavior_by_questionnaire/unadjusted_within_subject_results/min_obs_{min_obs}_min_unique_{min_unique}_min_obs_binary_{min_obs_per_level_when_binary}/{sub_id}'
            )
        else:
            contrast_map_dir = (
                cfg.output_root
                / f'within_subject_brain_behavior_by_questionnaire/within_subject_results/min_obs_{min_obs}_min_unique_{min_unique}_min_obs_binary_{min_obs_per_level_when_binary}/{sub_id}'
            )
        contrast_map_dir.mkdir(parents=True, exist_ok=True)
        nf.save(
            contrast_map,
            f'{contrast_map_dir}/{contrast}_behavioral_measures_effect_size_sub_{sub_id}.nii.gz',
        )

    # save design matrix
    design_matrix_path = contrast_map_dir / 'design_matrix.csv'
    design_matrix.to_csv(design_matrix_path, index=False)

    # save provenance
    log_provenance(contrast_map_dir)


def main():
    cfg = load_config()
    question_output_path = (
        cfg.output_root / 'within_subject_question_estimates/within_subject_results'
    )
    root = '/oak/stanford/groups/russpold/data/uh2/aim1'

    activation_maps = glob.glob(f'{question_output_path}/*/*.nii.gz')
    sub_ids = sorted(
        set([re.search('_sub_(.*).nii.gz', val).group(1) for val in activation_maps])
    )

    # Load events files
    events_files = sorted(
        glob.glob(f'{root}/BIDS/sub-s*/ses-[0-9]/func/*surveyMedley*modified*.tsv')
    )

    questionnaires = define_questionnaires()

    for sub_id in sub_ids:
        run_within_subject_questionnaire_glm(
            sub_id,
            events_files,
            questionnaires,
            question_output_path,
            activation_maps,
            min_obs=5,
            min_unique=3,
            min_obs_per_level_when_binary=None,
        )
        run_within_subject_questionnaire_glm(
            sub_id,
            events_files,
            questionnaires,
            question_output_path,
            activation_maps,
            min_obs=5,
            min_unique=2,
            min_obs_per_level_when_binary=2,
        )
        run_within_subject_questionnaire_glm(
            sub_id,
            events_files,
            questionnaires,
            question_output_path,
            activation_maps,
            min_obs=6,
            min_unique=3,
            min_obs_per_level_when_binary=None,
        )


if __name__ == '__main__':
    main()
