import argparse

from nilearn.glm.first_level import FirstLevelModel

from survey_medley_code.analysis_provenance import log_provenance
from survey_medley_code.config_loader import load_config
from survey_medley_code.within_subject_modeling.io_utils import (
    get_confounds_data,
    load_tsv_data,
)
from survey_medley_code.within_subject_modeling.timeseries_utils import (
    build_model_events,
    get_files,
    make_contrasts_question_estimates,
)


def main(subid):
    # config and file path fetcher
    cfg = load_config()
    files = get_files(cfg, subid)

    # build events df for model
    events = load_tsv_data(files['behav'])
    model_events = build_model_events(events)

    # pull out confounds
    confounds, percent_high_motion = get_confounds_data(files['confounds'])

    # stop if too much motion
    if percent_high_motion > 0.2:
        raise ValueError(
            f'Subject {subid} has too many high motion volumes ({percent_high_motion:.2f})'
        )

    # set up contrasts and output directory
    subid_no_s = subid.replace('s', '')
    contrast_dict = make_contrasts_question_estimates(events)
    within_subject_contrast_map_dir = (
        cfg.output_root
        / f'within_subject_question_estimates/within_subject_results/{subid_no_s}'
    )
    within_subject_contrast_map_dir.mkdir(parents=True, exist_ok=True)

    # run model
    tseries_model = FirstLevelModel(
        t_r=cfg.tr, hrf_model='spm', drift_model=None, smoothing_fwhm=cfg.smoothing_fwhm
    )
    tseries_model.fit(files['bold'], events=model_events, confounds=confounds)

    # estimate contrasts and save to disc

    for con_name, con in contrast_dict.items():
        filename = (
            within_subject_contrast_map_dir
            / f'contrast_{con_name}_effect_size_sub_{subid_no_s}.nii.gz'
        )
        con_est = tseries_model.compute_contrast(con, output_type='effect_size')
        con_est.to_filename(filename)

    # Save design matrix
    design_matrix = tseries_model.design_matrices_[0]
    design_matrix_path = within_subject_contrast_map_dir / 'design_matrix.csv'
    design_matrix.to_csv(design_matrix_path, index=False)

    # Save provenance
    log_provenance(within_subject_contrast_map_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run within-subject GLM for a single subject.'
    )
    parser.add_argument('subid', type=str, help="Subject ID (e.g., 's650')")
    args = parser.parse_args()
    main(args.subid)
