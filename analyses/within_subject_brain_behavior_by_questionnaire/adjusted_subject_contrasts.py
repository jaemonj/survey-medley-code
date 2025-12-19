#!/usr/bin/env python3

import glob
import re
from nilearn.glm.second_level import SecondLevelModel
import pandas as pd
import nibabel as nf

from survey_medley_code.config_loader import load_config


def get_sub_event_file(sub_id, events_files):
    for file in events_files:
        if sub_id in file:
            return file


def make_contrast_map(sub_id, events_files, questionnaires, question_output_path, activation_maps, cfg):
    sub_event_file = get_sub_event_file(sub_id, events_files)
    df = pd.read_csv(sub_event_file, sep='\t')
    # Get behavioral and bold data for each questionnaire, and a list of question lengths
    behavior_dict = {"grit": [], "brief": [], "future_time": [], "upps": [], "impulsive_venture": []}
    bold_dict = {"grit": [], "brief": [], "future_time": [], "upps": [], "impulsive_venture": []}
    q_len_list = []
    num_responses = 0
    for questionnaire in questionnaires:
        for question in questionnaires[questionnaire]:
            response = df.loc[df['trial_type'] == question, 'coded_response']
            behavior = response.values[0]
            text = df.loc[df['trial_type'] == question, 'item_text']
            chr_count = len(text.values[0])
            if not pd.isna(behavior):
                question_bold_file = f'{question_output_path}/{sub_id}/contrast_{question}_effect_size_sub_{sub_id}.nii.gz'
                if question_bold_file in activation_maps:
                    bold_dict[questionnaire].append(question_bold_file)
                    behavior_dict[questionnaire].append(behavior)
                    q_len_list.append(chr_count)
                    num_responses += 1
    # Make design matrix
    design_matrix = pd.DataFrame()
    # First columns are dummy regressors for each questionnaire with 3 or more responses
    start = 0
    for questionnaire in questionnaires:
        if(len(behavior_dict[questionnaire]) >= 3):
            one_indices = list(range(start, len(behavior_dict[questionnaire]) + start))
            start = one_indices[-1] + 1
            col = [1 if i in one_indices else 0 for i in range(num_responses)]
            design_matrix[questionnaire] = col
    # Other columns are questionnaire-specific behavioral regressors
    start = 0
    for questionnaire in questionnaires:
        if(len(behavior_dict[questionnaire]) >= 3):
            one_indices = list(range(start, len(behavior_dict[questionnaire]) + start))
            start = one_indices[-1] + 1
            col = [behavior_dict[questionnaire][i - one_indices[0]] if i in one_indices else 0 for i in range(num_responses)]
            design_matrix[f'{questionnaire}_responses'] = col
    # Add character count column to adjust for question length
    design_matrix['chr_count'] = q_len_list   
    
    cols_to_check = design_matrix.columns.drop('chr_count')
    mask = (design_matrix[cols_to_check] != 0).any(axis=1)
    design_matrix = design_matrix.loc[mask]
    removed_count = (~mask).sum()
    for i in range(removed_count):
        bold_dict["impulsive_venture"].pop(-1)

    # generate a contrast map for each questionnaire
    model = SecondLevelModel(n_jobs=2)
    sub_bold_files = []
    for questionnaire in questionnaires:
        sub_bold_files = sub_bold_files + bold_dict[questionnaire]
    model.fit(sub_bold_files, design_matrix=design_matrix)
    for i in range(int((design_matrix.shape[1] - 1) / 2)):
        contrast_map = model.compute_contrast(second_level_contrast=design_matrix.columns[i], output_type='effect_size')
        contrast_map_dir = (
            cfg.output_root
            / f'within_subject_brain_behavior_by_questionnaire/adjusted_within_subject_results/{sub_id}'
        )
        contrast_map_dir.mkdir(parents=True, exist_ok=True)
        nf.save(contrast_map, f'{contrast_map_dir}/{design_matrix.columns[i]}_behavioral_measures_effect_size_sub_{sub_id}.nii.gz')


def main():
    cfg = load_config()
    question_output_path = (
    cfg.output_root / 'within_subject_question_estimates/within_subject_results'
    )
    root = '/oak/stanford/groups/russpold/data/uh2/aim1'

    activation_maps = glob.glob(f'{question_output_path}/*/*.nii.gz')
    sub_ids = sorted(set([re.search('_sub_(.*).nii.gz', val).group(1) for val in activation_maps]))

    # Load events files
    events_files = sorted(glob.glob(f'{root}/BIDS/sub-s*/ses-[0-9]/func/*surveyMedley*modified*.tsv'))

    grit_questions = ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08"]
    brief_questions = ["Q09", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20", "Q21"]
    future_time_questions = ["Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30", "Q31"]
    upps_questions = ["Q32", "Q33", "Q34", "Q35", "Q36", "Q37"]
    impulsive_venture_questions = ["Q38", "Q39", "Q40"]
    questionnaires = {"grit": grit_questions, "brief": brief_questions, "future_time": future_time_questions, "upps": upps_questions, "impulsive_venture": impulsive_venture_questions}

    for sub in sub_ids:
        make_contrast_map(sub, events_files, questionnaires, question_output_path, activation_maps, cfg)


if __name__== '__main__':
    main()