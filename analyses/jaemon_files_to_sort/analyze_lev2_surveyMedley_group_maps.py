#!/usr/bin/env python3

import glob
import numpy as np
import nibabel as nf
import shutil
from pathlib import Path


if __name__ == "__main__":
    outdir = Path(f"/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/output_surveyMedley_noderivs_factors/"
                  f"surveyMedley_secondary_lev2_output/")
    outdir.mkdir(parents=True)

    lev2_out = ('/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/output_surveyMedley_noderivs_factors/'
                'surveyMedley_lev2_output')
    bold_files = sorted(
        glob.glob(f"{lev2_out}/*/*_contrast_*uncentered.nii.gz")
    )
    images = [nf.load(fname) for fname in bold_files]

    mean_outdir = Path(f"{outdir}/mean_maps")
    mean_outdir.mkdir(parents=True)

    for i in range(40):
        img_data = images[i].get_fdata()
        mean_data = np.mean(img_data, axis=3)
        images[i] = nf.Nifti1Image(mean_data, images[i].affine)
        nf.save(images[i], f"{mean_outdir}/Q{i + 1}_mean_activation.nii.gz")

    data = np.array([img.get_fdata() for img in images])

    factor_loadings_file = ('/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/utils_lev1'
                            '/surveyMedley_factor_loadings_oblimin.csv')
    factor_loadings = np.genfromtxt(factor_loadings_file, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5, 6))

    for f in range(6):
        contrast_weights = factor_loadings[:, f]
        contrast_weights = contrast_weights / np.sum(contrast_weights)
        weighted_avg = np.average(data, axis=0, weights=contrast_weights)

        se = np.sqrt(1 / np.sum(contrast_weights**2))

        z_stats = weighted_avg / se

        affine = images[0].affine
        nf.save(nf.Nifti1Image(weighted_avg, affine), f"{outdir}/factor_{f + 1}_weighted_avg.nii.gz")
        nf.save(nf.Nifti1Image(z_stats, affine), f"{outdir}/factor_{f + 1}_z_stats.nii.gz")




