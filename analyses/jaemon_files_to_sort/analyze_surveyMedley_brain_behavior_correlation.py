import os
import glob
import re
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Optional
import nibabel as nib
import numpy as np


def create_design_matrix(events_files, question_num):
    response_data = []
    for file in events_files:
        df = pd.read_csv(file, sep='\t')
        response = df.loc[df['trial_type'] == 'Q' + question_num, 'coded_response']
        behavior = response.values[0]
        response_data.append({'behavior': behavior, 'intercept': 1})

    design_matrix = pd.DataFrame(response_data)
    return design_matrix


def second_level_model(bold_files, question_num, events_files):
    # extract only the files for the question_num
    question_bold_files = [file for file in bold_files if 'Q' + question_num in file]

    sub_list = [re.search('_sub_(.*)_rtmodel_', val).group(1) for val in question_bold_files]
    print(sub_list)

    # remove outlier subjects and subjects who didn't answer the question
    question_events_files = [file for file in events_files if re.search(r'\d+', os.path.basename(file)).group(0) in sub_list]
    print(question_events_files)
    
    design_matrix = create_design_matrix(question_events_files, question_num)
    model = SecondLevelModel(n_jobs=2)
    model.fit(question_bold_files, design_matrix=design_matrix)
    z_map = model.compute_contrast('behavior', output_type='z_score')
    return z_map


def compute_colorbar_range(
    image_paths: List[str], percentile: float = 99
) -> tuple[float, float]:
    """Compute symmetric colorbar limits based on pooled voxel data across images."""
    all_data = []
    for path in image_paths:
        data = nib.load(path).get_fdata()
        all_data.append(data[np.isfinite(data)])  # ignore NaNs/Infs
    all_values = np.concatenate(all_data)
    vmax = np.percentile(np.abs(all_values), percentile)
    return -vmax, vmax


def main():
    root = '/oak/stanford/groups/russpold/data/uh2/aim1'
    events_files = sorted(
        glob.glob(f'{root}/BIDS/sub-s*/ses-[0-9]/func/*surveyMedley*modified*.tsv')
    )
    lev1_out = (f'{root}/derivatives/output_surveyMedley_noderivs_factors/'
                f'surveyMedley_lev1_output/task_surveyMedley_rtmodel_rt_uncentered')
    bold_files = sorted(
        glob.glob(f'{lev1_out}/contrast_estimates/*_contrast_Q*rtmodel_rt_uncentered*')
    )

    z_slices = [-20, -10, 0, 10, 20, 30, 40, 50]
    vmin, vmax = compute_colorbar_range(bold_files, percentile=99)

    # run the second level model for each question and add the zstat maps to the pdf
    with PdfPages('brain_behavior_zstat_maps.pdf') as pdf:
        for i in range(40):
            question_num = str((i + 1)).zfill(2)
            z_map = second_level_model(bold_files, question_num, events_files)
            plotting.plot_stat_map(
                z_map,
                display_mode='z',
                cut_coords=z_slices,
                threshold=None,
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
                title=f'Question {question_num} unthresholded'
            )
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)

            plotting.plot_stat_map(
                z_map,
                display_mode='z',
                cut_coords=z_slices,
                threshold=2.0,
                vmin=vmin,
                vmax=vmax,
                colorbar=True,
                title=f'Question {question_num} thresholded'
            )
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    main()