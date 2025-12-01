import glob
import json
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np
import pandas as pd

from survey_medley_code.within_subject_modeling.io_utils import (
    resolve_file,
)

# from utils_lev1.qa import (
#     add_to_html_summary,
#     qa_design_matrix,
#     update_excluded_subject_csv,
# )


def get_confounds_data(confounds_file):
    """
    Creates nuisance regressors for output from fmriprep.
    input:
        confounds_file: path to confounds file from fmriprep
    output:
        confound_regressors: includes motion regressors, FD, nonsteady volumes,
            cosine basis set
        percent_high_motion:  Percentage of high motion time points.  High motion
            is defined by the following
            FD>.5, stdDVARS>1.2 (that relates to DVARS>.5)
    """
    confounds_df = pd.read_csv(confounds_file, sep='\t', na_values=['n/a']).fillna(0)
    excessive_movement = (confounds_df.framewise_displacement > 0.5) | (
        confounds_df.std_dvars > 1.2
    )
    percent_high_motion = np.mean(excessive_movement)
    confounds = confounds_df.filter(
        regex='^(cosine00|cosine01|cosine02|trans|rot)'
    ).copy()
    confounds = confounds.loc[:, ~confounds.columns.str.endswith('power2')]

    return confounds, percent_high_motion


def get_nscans(timeseries_data_file):
    """
    Get the number of time points from 4D data file
    input: time_series_data_file: Path to 4D file
    output: nscans: number of time points
    """
    import nibabel as nb

    fmri_data = nb.load(timeseries_data_file)
    n_scans = fmri_data.shape[3]
    return n_scans


def get_tr(root, task):
    """
    Get the TR from the bold json file
    input:
        root: Root for BIDS data directory
        task: Task name
    output: TR as reported in json file (presumable in s)
    """
    json_file = glob.glob(f'{root}/BIDS/*{task}_bold.json')[0]
    with open(json_file, 'rb') as f:
        task_info = json.load(f)
    tr = task_info['RepetitionTime']
    return tr


def make_desmat_contrasts(
    root, task, events_file, add_deriv, n_scans, confounds_file=None, regress_rt='no_rt'
):
    """
    Creates design matrices and contrasts for each task.  Should work for any
    style of design matrix as well as the regressors are defined within
    the imported make_task_desmat_fcn_map (dictionary of functions).
    A single RT regressor can be added using regress_rt='rt_uncentered'
    Input:
        root:  Root directory (for BIDS data)
        task: Task name
        events_file: File path to events.tsv for the given task
        add_deriv: 'deriv_yes' or 'deriv_no', recommended to use 'deriv_yes'
        n_scans: Number of scans
        confound_file (optional): File path to fmriprep confounds file
        regress_rt: 'no_rt' or 'rt_uncentered' or 'rt_centered' or 'rt_duration' ('rt_duration' only for surveyMedley)
    Output:
        design_matrix, contrasts: Full design matrix and contrasts for nilearn model
        percent junk: percentage of trials labeled as "junk".  Used in later QA.
        percent high motion: percentage of time points that are high motion.  Used later in QA.
    """
    from utils_lev1.first_level_designs_including_survey_medley import (
        make_task_desmat_fcn_dict,
    )

    if confounds_file is not None:
        confound_regressors, percent_high_motion = get_confounds_data(confounds_file)
    else:
        confound_regressors = None

    tr = get_tr(root, task)

    design_matrix, contrasts, percent_junk, events_df = make_task_desmat_fcn_dict[task](
        events_file, add_deriv, regress_rt, n_scans, tr, confound_regressors
    )
    return design_matrix, contrasts, percent_junk, percent_high_motion, tr, events_df


def check_file(glob_out):
    """
    Checks if file exists
    input:
        glob_out: output from glob call attempting to retreive files.  Note this
        might be simplified for other data.  Since the tasks differed between sessions
        across subjects, the ses directory couldn't be hard coded, in which case glob
        would not be necessary.
    output:
        file: Path to file, if it exists
        file_missing: Indicator for whether or not file exists (used in later QA)
    """
    if len(glob_out) > 0:
        file = glob_out[0]
        file_missing = [0]
    else:
        file = []
        file_missing = [1]
    return file, file_missing


# This currently won't work as-is...Let me know if you need it and I'll fix it.
# def get_files(root, subid, task):
#     """Fetches files (events.tsv, confounds, mask, data)
#     if files are not present, excluded_subjects.csv is updated and
#     program exits
#     input:
#         root:  Root directory
#         subid: subject ID (without s prefix)
#         task: Task
#     output:
#        files: Dictionary with file paths (or empty lists).  Needs to be further
#            processed by check_file() to pick up instances when task is not available
#            for a given subject (missing data files)
#            Dictionary contains events_file, mask_file, confounds_file, data_file
#     """
#     files = {}
#     file_missing = {}
#     file_missing['subid_task'] = f'{subid}_{task}'
#     # files['events_file'], file_missing['event_file_missing'] = check_file(glob.glob(
#     #     f'{root}/BIDS/sub-s{subid}/ses-[0-9]/func/*{task}*tsv'
#     # ))
#     files['events_file'], file_missing['event_file_missing'] = check_file(
#         glob.glob(f'{root}/BIDS/sub-s{subid}/ses-[0-9]/func/*{task}*modified*.tsv')
#     )
#     files['confounds_file'], file_missing['confounds_file_missing'] = check_file(
#         glob.glob(
#             f'{root}/derivatives/fmriprep/sub-s{subid}/ses-[0-9]/func/*{task}*confounds*.tsv'
#         )
#     )

#     files['mask_file'], file_missing['mask_file_missing'] = check_file(
#         glob.glob(
#             f'{root}/derivatives/fmriprep/sub-s{subid}/ses-[0-9]/func/*{task}*space-MNI152NLin2009cAsym*mask*.nii.gz'
#         )
#     )
#     files['data_file'], file_missing['data_file_missing'] = check_file(
#         glob.glob(
#             # f'{root}/derivatives/fmriprep/sub-s{subid}/ses-[0-9]/func/*{task}*AROMA*_bold.nii.gz'
#             f'{root}/derivatives/fmriprep/sub-s{subid}/ses-[0-9]/func/*{task}*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
#         )
#     )
#     file_missing = pd.DataFrame(file_missing)
#     if (
#         file_missing.loc[:, file_missing.columns != 'subid_task']
#         .gt(0)
#         .any(axis=1)
#         .bool()
#     ):
#         update_excluded_subject_csv(file_missing, subid, task, contrast_dir)
#         print(f'Subject {subid}, task: {task} is missing one or more input data files.')
#         sys.exit(0)
#     return files


def get_files(cfg, subid):
    files = {}
    for ft in ['bold', 'mask', 'behav', 'confounds']:
        try:
            files[ft] = resolve_file(cfg, subid, ft)
        except ValueError as e:
            raise RuntimeError(f'{ft} file missing for sub-{subid}') from e
    return files


def build_model_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Given a BIDS-style events DataFrame with columns:
        - onset
        - duration
        - trial_type
        - coded_response (NaN for errors)
        - response_time (NaN when no RT)
    Returns a dataframe with:
        - trial_type (with 'err_' prefix for error trials)
        - RT regressor as its own condition ('rt')
    """

    # Work on a copy
    events = events.copy()

    # --- Create revised trial type ---
    events['revised_trial_type'] = events['trial_type']
    error_mask = events['coded_response'].isna()
    events.loc[error_mask, 'revised_trial_type'] = (
        'err_' + events.loc[error_mask, 'trial_type']
    )

    # --- Prepare base model events ---
    model_events = (
        events[['onset', 'duration', 'revised_trial_type']]
        .rename(columns={'revised_trial_type': 'trial_type'})
        .copy()
    )

    # --- Build RT regressor ---
    rt_events = (
        events[['onset', 'response_time']]
        .dropna(subset=['response_time'])
        .rename(columns={'response_time': 'duration'})
        .copy()
    )
    rt_events['trial_type'] = 'rt'

    # --- Concatenate model & RT regressors ---
    final = pd.concat([model_events, rt_events], ignore_index=True)

    # --- Sort by onset ---
    final = final.sort_values('onset').reset_index(drop=True)

    return final


def make_contrasts_question_estimates(events: pd.DataFrame):
    """
    Create a dictionary of contrasts for nilearn second-level model.

    1. One contrast per unique trial_type (key=value)
    2. One contrast per unique survey:
       key = survey name
       value = weighted string average of trial_types belonging to that survey
    Trials with missing coded_response get 'err_' prepended (to match design matrix columns).
    """
    contrasts = {}

    # --- Add revised trial_type for missing responses ---
    events = events.copy()
    events['revised_trial_type'] = events['trial_type']
    events.loc[events['coded_response'].isna(), 'revised_trial_type'] = (
        'err_' + events.loc[events['coded_response'].isna(), 'trial_type']
    )

    # --- Trial type contrasts ---
    trial_types = events['revised_trial_type'].unique()
    for tt in trial_types:
        contrasts[tt] = tt

    # --- Survey contrasts ---
    surveys = events['survey'].unique()
    for survey in surveys:
        tt_for_survey = events.loc[
            events['survey'] == survey, 'revised_trial_type'
        ].unique()
        n_tt = len(tt_for_survey)
        # build string like '1/3*Q01 + 1/3*Q05 + 1/3*Q10'
        terms = [f'1/{n_tt}*{tt}' for tt in tt_for_survey]
        contrast_string = ' + '.join(terms)
        contrasts[survey] = contrast_string

    return contrasts
