import glob
import pandas as pd
from nilearn import masking, plotting
import re
import numpy as np
from collections import OrderedDict
import nibabel as nib
from scipy.stats import spearmanr
from nilearn.maskers import NiftiMasker
import mantel
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_fmri_dict(sub_ids, complete_data, group_mask_img):
    """
        Create a memory-efficient dictionary of fMRI data using NiftiMasker.
        Files are matched explicitly by Q01–Q40 substrings to ensure order.

        Parameters
        ----------
        sub_ids : list of str
            Subject IDs to extract from complete_data.
        complete_data : list of str
            List of file paths to 3D fMRI NIfTI images.
        group_mask_img : nibabel.Nifti1Image
            Mask image already loaded into memory.

        Returns
        -------
        data_dict : OrderedDict
            Dictionary where keys are subject IDs and values are (40, n_voxels) arrays.
        """
    masker = NiftiMasker(mask_img=group_mask_img)
    data_dict = OrderedDict()

    for sub_id in sub_ids:
        # Select files for this subject
        sub_files = [f for f in complete_data if sub_id in f]

        # Ensure 1 file for each Q01–Q40
        ordered_paths = []
        for q in range(1, 41):
            qstr = f'Q{q:02d}'
            matches = [f for f in sub_files if qstr in f]
            if len(matches) != 1:
                raise ValueError(
                    f'Expected one match for {qstr} in subject {sub_id}, found {len(matches)}'
                )
            ordered_paths.append(matches[0])

        # Load and mask
        imgs = [nib.load(p) for p in ordered_paths]
        stacked_img = nib.concat_images(imgs)
        masked_data = masker.fit_transform(stacked_img)  # shape: (40, n_voxels)

        data_dict[sub_id] = masked_data

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


def fast_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    sum_x = x.sum()
    sum_y = y.sum()
    sum_x2 = np.dot(x, x)
    sum_y2 = np.dot(y, y)
    sum_xy = np.dot(x, y)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

    return numerator / denominator if denominator != 0 else 0.0


def get_bold_dist_mat(fmri_matrix, voxel_index):
    """
    Computes BOLD correlation distances at a given voxel across subjects.

    Parameters:
    - fmri_matrix: shape (n_subjects, n_timepoints, n_voxels)
    - voxel_index: integer index into the 3rd dimension of fmri_matrix

    Returns:
    - bold_dist_mat_full: full symmetric distance matrix
    - bold_dist: BOLD correlation distances in vector form
    """
    n_subs = fmri_matrix.shape[0]
    bold_dist_mat = np.zeros((n_subs, n_subs))
    bold_dist = []

    # Compute upper triangle of distance matrix
    for i in range(n_subs):
        ts_i = fmri_matrix[i, :, voxel_index]
        for j in range(i + 1, n_subs):
            ts_j = fmri_matrix[j, :, voxel_index]
            r = fast_pearson(ts_i, ts_j)
            dist = 1 - r
            bold_dist_mat[i, j] = dist
            bold_dist.append(dist)

    # Symmetrize matrix
    bold_dist_mat_full = bold_dist_mat + bold_dist_mat.T

    return bold_dist_mat_full, bold_dist


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
    # also exclude the subjects contributing to the most dropout
    excluded_subject_ids.add('445')
    excluded_subject_ids.add('234')

    subject_masks = [mask for mask in subject_masks if not any(id in mask for id in excluded_subject_ids)]

    group_mask = masking.intersect_masks(subject_masks, threshold=1)

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
    sorted_sub_ids = [subid for subid in sorted_sub_ids if subid not in excluded_subject_ids]

    # Step 3: Load and store the contrast maps
    complete_data = sorted([file for file in bold_files if any(sub_id in file for sub_id in sorted_sub_ids)])
    fmri_data_dict = create_fmri_dict(sorted_sub_ids, complete_data, group_mask)

    # Step 4: Estimate behavioral similarity
    events_files = sorted(glob.glob(f'{root}/BIDS/sub-s*/ses-[0-9]/func/*surveyMedley*modified*.tsv'))
    n_subs = len(sorted_sub_ids)
    # make a dictionary for the vectors of the behavioral responses of each subject
    behav_data_dict = create_behav_dict(sorted_sub_ids, events_files)

    behav_dist = []
    behav_dist_mat = np.zeros((n_subs, n_subs))
    for i in range(n_subs):
        behav_sub_i = behav_data_dict[sorted_sub_ids[i]]
        for j in range(i + 1, n_subs):
            behav_sub_j = behav_data_dict[sorted_sub_ids[j]]
            r = fast_pearson(behav_sub_i, behav_sub_j)
            behav_dist.append(1 - r)
            behav_dist_mat[i, j] = 1 - r
    behav_dist = np.array(behav_dist)
    behav_dist_mat_full = behav_dist_mat + behav_dist_mat.T
    np.fill_diagonal(behav_dist_mat_full, 0)

    # Step 5: Compute pairwise similarity for each voxel and correlate with behavioral similarity
    fmri_matrix = np.stack(
        [fmri_data_dict[sub_id] for sub_id in sorted_sub_ids]
    )  # shape: (n_subs, n_timepoints, n_voxels)
    # to be safe, calculate n_voxels now
    first_key = next(iter(fmri_data_dict))
    n_voxels = fmri_data_dict[first_key].shape[1]

    spearman_corrs = np.zeros(n_voxels)
    p_vals = np.zeros(n_voxels)

    # for each voxel, calculate Spearman correlations and Mantel test-derived p-values
    for v in range(n_voxels):
        bold_dist_mat_full, bold_dist = get_bold_dist_mat(fmri_matrix, v)
        rho, _ = spearmanr(behav_dist, bold_dist)
        spearman_corrs[v] = rho
        result = mantel.test(bold_dist_mat_full, behav_dist_mat_full, perms=10000, method='spearman', tail='two-tail')
        p_vals[v] = result.p

    group_mask_data = group_mask.get_fdata().astype(bool)

    # Save out Spearman correlations
    output_data = np.zeros(group_mask_data.shape)
    output_data[group_mask_data] = spearman_corrs
    spearman_img = nib.Nifti1Image(output_data, affine=group_mask.affine, header=group_mask.header)
    nib.save(spearman_img, 'spearman_correlations.nii.gz')

    # Perform FDR correction
    fdr_results = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    # fdr_results[0] is a boolean array indicating which p-values are significant
    significant = fdr_results[0]
    significant_data = np.zeros(group_mask_data.shape)
    significant_data[group_mask_data] = significant.astype(np.uint8)
    significant_map = nib.Nifti1Image(significant_data, affine=group_mask.affine, header=group_mask.header)
    nib.save(significant_map, 'FDR_adjusted_significant_p_values.nii.gz')

    # fdr_results[1] is the adjusted p-values (q-values)
    qvals = fdr_results[1]
    qval_data = np.zeros(group_mask_data.shape)
    qval_data[group_mask_data] = 1 - qvals
    qval_map = nib.Nifti1Image(qval_data, affine=group_mask.affine, header=group_mask.header)
    nib.save(qval_map, 'FDR_adjusted_1_minus_p_values.nii.gz')

    # Get the thresholded Spearman correlation map
    thresholded_correlation_data = significant_data * output_data
    thresholded_correlation_map = nib.Nifti1Image(thresholded_correlation_data, affine=group_mask.affine, header=group_mask.header)
    nib.save(thresholded_correlation_map, 'significant_spearman_correlations.nii.gz')

    # Plot some figures
    pval_data = np.zeros(group_mask_data.shape)
    pval_data[group_mask_data] = 1 - p_vals
    pval_map = nib.Nifti1Image(pval_data, affine=group_mask.affine, header=group_mask.header)

    z_slices = [-20, -10, 0, 10, 20, 30, 40, 50]
    thresh = 0.95
    with PdfPages('Mantel_test_whole_brain_figures.pdf') as pdf:
        plotting.plot_stat_map(
            pval_map,
            display_mode='z',
            cut_coords=z_slices,
            threshold=thresh,
            vmin=0.95,
            vmax=1.0,
            title=f'1 - p-values before FDR correction (threshold: 1-p_value > {thresh})'
        )
        fig = plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)
        plotting.plot_stat_map(
            qval_map,
            display_mode='z',
            cut_coords=z_slices,
            threshold=thresh,
            vmin=0.95,
            vmax=1.0,
            title=f'FDR adjusted 1 - p-values (threshold: 1-p_value > {thresh})'
        )
        fig = plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)
        plotting.plot_stat_map(
            spearman_img,
            display_mode='z',
            cut_coords=z_slices,
            threshold=None,
            title='Unthresholded Spearman correlations',
        )
        fig = plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)
        plotting.plot_stat_map(
            thresholded_correlation_map,
            display_mode='z',
            cut_coords=z_slices,
            threshold=None,
            title='Significant Spearman correlations',
        )
        fig = plt.gcf()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == '__main__':
    main()