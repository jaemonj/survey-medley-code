#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
from nilearn.glm.second_level import SecondLevelModel

from survey_medley_code.config_loader import load_config
from survey_medley_code.within_subject_modeling.timeseries_utils import get_files


def load_data(cfg):
    output_root = (
        cfg.output_root
        / 'within_subject_adjusted_questionnaire_averages/within_subject_results'
    )
    question_output_path = (
        cfg.output_root / 'within_subject_question_estimates/within_subject_results'
    )
    outlier_assessment = question_output_path / 'outlier_assessment'

    sub_ids, bold_paths, question_names_all = [], [], []
    question_names = [f'Q{i:02d}' for i in range(1, 41)]

    for question_name in question_names:
        good_sub_file = (
            outlier_assessment
            / f'subjects_outlier_percent_lt_8_contrast_{question_name}.txt'
        )
        good_subs_list = good_sub_file.read_text().splitlines()
        for good_sub in good_subs_list:
            sub_no_s = good_sub.replace('s', '')
            bold_paths_loop = list(
                question_output_path.glob(
                    f'{sub_no_s}/contrast_*{question_name}_effect_size_sub_{sub_no_s}.nii.gz'
                )
            )
            if bold_paths_loop:
                sub_ids.append(sub_no_s)
                question_names_all.append(question_name)
                bold_paths.append(bold_paths_loop[0])
            else:
                print(
                    f'Output file missing for question/subject: {question_name}/{good_sub}'
                )

    return sub_ids, bold_paths, question_names_all, output_root


def prepare_events(cfg, sub_id):
    files = get_files(cfg, f's{sub_id}')
    events = pd.read_csv(files['behav'], sep='\t')
    events['chr_count'] = events['item_text'].str.len()
    mean_chr_count = events['chr_count'].mean()
    events['chr_count_centered'] = events['chr_count'] - mean_chr_count
    events = events.sort_values('trial_type')
    return events[['chr_count', 'chr_count_centered', 'survey', 'trial_type']]


def get_subject_design_matrix(merged_data, survey_dummies, subject_id):
    subject_data = merged_data[merged_data['sub_id'] == subject_id]
    design_matrix = subject_data[
        ['bold_paths', 'chr_count_centered'] + list(survey_dummies.columns)
    ]
    for col in survey_dummies.columns:
        if design_matrix[col].sum() == 0:
            print(
                f'Removing column {col} for subject {subject_id} as it contains all zeros.'
            )
            design_matrix = design_matrix.drop(col, axis=1)
    return design_matrix


def run_second_level_model(subject_id, output_root, merged_data, survey_dummies):
    design_matrix = get_subject_design_matrix(merged_data, survey_dummies, subject_id)
    bold_paths = design_matrix['bold_paths'].tolist()
    X = design_matrix.drop('bold_paths', axis=1)

    # Save design matrix as CSV
    sub_output = output_root / subject_id
    sub_output.mkdir(parents=True, exist_ok=True)
    csv_file = sub_output / f'desmat_sub_{subject_id}.csv'
    X.to_csv(csv_file, index=False)
    print(f'Saved design matrix to {csv_file}')

    second_level_model = SecondLevelModel()
    second_level_model.fit(bold_paths, design_matrix=X)

    survey_columns = [col for col in X.columns]
    print(survey_columns)

    for survey in survey_columns:
        contrast = second_level_model.compute_contrast(
            survey, output_type='effect_size'
        )
        output_file = (
            sub_output / f'contrast_{survey}_effect_size_sub_{subject_id}.nii.gz'
        )
        contrast.to_filename(str(output_file))
        print(f'Saved contrast for {survey} to {output_file}')


def main(job_num):
    cfg = load_config()
    sub_ids, bold_paths, question_names_all, output_root = load_data(cfg)
    events = prepare_events(cfg, sub_ids[0])

    bold_data = pd.DataFrame(
        {'sub_id': sub_ids, 'bold_paths': bold_paths, 'trial_type': question_names_all}
    )
    merged_data = pd.merge(bold_data, events, on='trial_type', how='left')
    survey_dummies = pd.get_dummies(merged_data['survey'], prefix='survey').astype(int)
    survey_dummies.columns = survey_dummies.columns.str.replace('survey_', '')
    merged_data = pd.concat([merged_data, survey_dummies], axis=1)

    unique_subids = merged_data['sub_id'].unique()
    if job_num < 0 or job_num >= len(unique_subids):
        print(f'Invalid job number. Must be between 0 and {len(unique_subids) - 1}')
        sys.exit(1)

    subject_id = unique_subids[job_num]
    run_second_level_model(subject_id, output_root, merged_data, survey_dummies)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python script_name.py <job_number>')
        sys.exit(1)

    try:
        job_num = int(sys.argv[1])
        main(job_num)
    except ValueError:
        print('Job number must be an integer')
        sys.exit(1)
