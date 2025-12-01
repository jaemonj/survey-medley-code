from survey_medley_code.outlier_detection import generate_all_data_summaries
import glob
import re
from pathlib import Path 
from collections import defaultdict

data_output_dir = Path(
    '/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_by_questionnaire/'
)

# --- collect all files ---
all_files = sorted(data_output_dir.glob('*effect_size*.nii.gz'))

# --- use regex to extract contrast name and subject id ---
pattern = re.compile(r'(.+?)_behavioral_measures_effect_size_sub_(\d+).nii.gz')

# group by contrast
contrast_groups = defaultdict(lambda: {'nifti_paths': [], 'image_labels': []})

for f in all_files:
    match = pattern.search(f.name)
    if match:
        contrast_name, subj_id = match.groups()
        contrast_groups[contrast_name]['nifti_paths'].append(f)
        contrast_groups[contrast_name]['image_labels'].append(subj_id)
    else:
        print(f'⚠️ Skipping file (pattern not matched): {f.name}')

# --- build the final list of dictionaries ---
contrast_dicts = []
for contrast_name, info in contrast_groups.items():
    contrast_dicts.append(
        {
            'main_title': contrast_name,
            'nifti_paths': info['nifti_paths'],
            'image_labels': info['image_labels'],
            'data_type_label': 'contrast estimates',
        }
    )

# --- optional: inspect one example ---
print(f'Found {len(contrast_dicts)} contrasts')
print(contrast_dicts[0])


generate_all_data_summaries(
    contrast_dicts,
    n_std=3,
    output_dir='/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_by_questionnaire/behavioral_measures_effect_size_outlier_analysis')