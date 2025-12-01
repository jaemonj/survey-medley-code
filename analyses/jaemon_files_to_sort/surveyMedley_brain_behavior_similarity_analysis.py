import glob
import pandas as pd
from nilearn import masking
import re
import numpy as np
from collections import OrderedDict
import nibabel as nib
from scipy.stats import spearmanr


def create_fmri_dict(sub_ids, complete_data, mask_data, n_voxels):
    data_dict = OrderedDict()
    for sub_id in sub_ids:
        subject_array = np.zeros((40, n_voxels))
        sub_data = sorted([file for file in complete_data if sub_id in file])
        for i in range(40):
            img = nib.load(sub_data[i])
            data = img.get_fdata()
            masked_values = data[mask_data]
            subject_array[i, :] = masked_values
        data_dict[sub_id] = subject_array
    return data_dict


def get_sub_event_file(sub_id, events_files):
    for file in events_files:
        if sub_id in file:
            return file


def create_behav_dict(sub_ids, events_files):
    behav_dict = {}
    for sub_id in sub_ids:
        behav_vector = []
        sub_event_file = get_sub_event_file(sub_id, events_files)
        df = pd.read_csv(sub_event_file, sep='\t')
        # create behav_vector, making sure the responses are in order from Q01-40
        for i in range(40):
            str_question_num = 'Q' + str(i + 1).zfill(2)
            response = df.loc[df['trial_type'] == str_question_num, 'coded_response']
            behavior = response.values[0]
            behav_vector.append(behavior)
        behav_dict[sub_id] = np.array(behav_vector)
    return behav_dict


def main():
    root = '/oak/stanford/groups/russpold/data/uh2/aim1'
    lev1_out = (f'{root}/derivatives/output_surveyMedley_noderivs_factors/surveyMedley_lev1_output'
                f'/task_surveyMedley_rtmodel_rt_uncentered')
    bold_files = sorted(glob.glob(f'{lev1_out}/contrast_estimates/*Q*'))

    # Step 1: Make a group mask
    subject_masks = glob.glob(f'{root}/derivatives/fmriprep/sub-s*/ses-['
                              f'0-9]/func/*surveyMedley*space-MNI152NLin2009cAsym*mask*.nii.gz')
    # some of the subject masks are of subjects excluded from all analysis, so remove these masks
    excluded_subjects_csv = pd.read_csv(f'{lev1_out}/excluded_subject.csv')
    excluded_subjects_column = excluded_subjects_csv['subid_task'].tolist()
    excluded_subject_ids = set([s[:3] for s in excluded_subjects_column])
    subject_masks = [mask for mask in subject_masks if not any(id in mask for id in excluded_subject_ids)]

    group_mask = masking.intersect_masks(subject_masks)

    mask_data = group_mask.get_fdata().astype(bool)
    n_voxels = np.sum(mask_data)

    # Step 2: Sorted list of subject IDs for subjects who answered all questions
    all_sub_ids = set([re.search('_sub_(.*)_rtmodel_', val).group(1) for val in bold_files])
    sub_ids = []
    for sub in all_sub_ids:
        all_questions_present = True
        for i in range(40):
            str_question_num = 'Q' + str(i + 1).zfill(2)
            if f'{lev1_out}/contrast_estimates/task_surveyMedley_contrast_{str_question_num}_v_baseline_sub_{sub}_rtmodel_rt_uncentered_stat_contrast.nii.gz' not in bold_files:
                all_questions_present = False
        if all_questions_present:
            sub_ids.append(sub)
    sorted_sub_ids = sorted(sub_ids)

    # Step 3: Load and store the contrast maps
    complete_data = sorted([file for file in bold_files if any(sub_id in file for sub_id in sorted_sub_ids)])
    fmri_data_dict = create_fmri_dict(sorted_sub_ids, complete_data, mask_data, n_voxels)

    # Step 4: Estimate behavioral similarity
    events_files = sorted(glob.glob(f'{root}/BIDS/sub-s*/ses-[0-9]/func/*surveyMedley*modified*.tsv'))
    n_subs = len(sorted_sub_ids)
    # make a dictionary for the vectors of the behavioral responses of each subject
    behav_data_dict = create_behav_dict(sorted_sub_ids, events_files)
    
    df = pd.DataFrame(columns=sorted_sub_ids, index=sorted_sub_ids)
    behav_dist = []
    for i in range(n_subs):
        behav_sub_i = behav_data_dict[sorted_sub_ids[i]]
        for j in range(i + 1, n_subs):
            behav_sub_j = behav_data_dict[sorted_sub_ids[j]]
            r = np.corrcoef(behav_sub_i, behav_sub_j)
            behav_dist.append(1 - r[0, 1])
            df.loc[sorted_sub_ids[i], sorted_sub_ids[j]] = 1 - r[0, 1]
    df.to_csv('behavior_correlation_distance_matrix.csv')
    behav_dist = np.array(behav_dist)

    # Step 5: Compute pairwise similarity for each voxel and correlate with behavioral similarity
    spearman_corrs = np.zeros(n_voxels)
    for v in range(n_voxels):
        voxel_dist = []
        for i in range(n_subs):
            fmri_sub_i = fmri_data_dict[sorted_sub_ids[i]]
            voxel_sub_i = fmri_sub_i[:, v]
            for j in range(i + 1, n_subs):
                fmri_sub_j = fmri_data_dict[sorted_sub_ids[j]]
                voxel_sub_j = fmri_sub_j[:, v]
                r = np.corrcoef(voxel_sub_i, voxel_sub_j)
                voxel_dist.append(1 - r[0, 1])
        rho, _ = spearmanr(behav_dist, voxel_dist)
        spearman_corrs[v] = rho
    output_data = np.zeros(mask_data.shape)
    output_data[mask_data] = spearman_corrs
    spearman_img = nib.Nifti1Image(output_data, affine=group_mask.affine, header=group_mask.header)
    nib.save(spearman_img, 'spearman_correlations.nii.gz')


if __name__ == '__main__':
    main()
