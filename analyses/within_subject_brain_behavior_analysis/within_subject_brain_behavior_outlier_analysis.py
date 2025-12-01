import glob
import re
from survey_medley_code.outlier_detection import generate_all_data_summaries


def main():
    root = '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_analysis'
    nifti_paths = sorted(glob.glob(f'{root}/behavioral_measures*.nii.gz'))
    sub_ids = sorted(set([re.search('_sub_(.*).nii.gz', val).group(1) for val in nifti_paths]))
    data_dict = {
        'main_title': 'behavioral_measures_effect_size',
        'nifti_paths': nifti_paths,
        'image_labels': sub_ids, 
        'data_type_label': 'Contrast Estimate'
    }
    dict_list = [data_dict]
    generate_all_data_summaries(dict_list, n_std=3, output_dir='/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_analysis/outlier_analysis')


if __name__ == '__main__':
    main()