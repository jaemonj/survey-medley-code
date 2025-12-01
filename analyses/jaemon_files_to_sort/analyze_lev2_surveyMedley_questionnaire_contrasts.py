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


def second_level_model(bold_files, contrast):
    # extract only the files for the contrast
    contrast_bold_files = [file for file in bold_files if contrast in file]
    
    design_matrix = pd.DataFrame(
        [1] * len(contrast_bold_files),
        columns=['intercept'],
    )
    model = SecondLevelModel(n_jobs=2)
    model.fit(contrast_bold_files, design_matrix=design_matrix)
    z_map = model.compute_contrast('intercept', output_type='z_score')
    return z_map


def main():
    root = '/oak/stanford/groups/russpold/data/uh2/aim1'
    lev1_out = (f'{root}/derivatives/output_surveyMedley_noderivs_questionnaire_contrasts/'
                f'surveyMedley_lev1_output/task_surveyMedley_rtmodel_rt_uncentered')
    bold_files = sorted(
        glob.glob(f'{lev1_out}/contrast_estimates/*')
    )
    contrasts = ['grit_v_brief', 'grit_v_future_time', 'grit_v_upps', 'brief_v_future_time', 'brief_v_upps', 'future_time_v_upps', 'grit_v_impulsive_venture', 'brief_v_impulsive_venture', 'future_time_v_impulsive_venture', 'upps_v_impulsive_venture']
    
    z_slices = [-20, -10, 0, 10, 20, 30, 40, 50]

    # run the second level model for each question and add the zstat maps to the pdf
    with PdfPages('questionnaire_zstat_maps.pdf') as pdf:
        for contrast in contrasts:
            z_map = second_level_model(bold_files, contrast)
            print(f'{contrast} z_map min: {np.min(z_map)}, max: {np.max(z_map)}')
            plotting.plot_stat_map(
                z_map,
                display_mode='z',
                cut_coords=z_slices,
                threshold=None,
                vmin=-4.0,
                vmax=4.0,
                colorbar=True,
                title=f'{contrast} unthresholded'
            )
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)
            
            plotting.plot_stat_map(
                z_map,
                display_mode='z',
                cut_coords=z_slices,
                threshold=2.0,
                vmin=-4.0,
                vmax=4.0,
                colorbar=True,
                title=f'{contrast} thresholded'
            )
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    main()
