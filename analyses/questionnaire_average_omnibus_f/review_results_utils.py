from itertools import combinations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn.plotting import plot_stat_map
from scipy import ndimage


def get_p_t_maps(outdir):
    questionnaire_names = ['brief', 'future_time', 'grit', 'impulsive_venture', 'upps']
    question_pmaps = {}
    question_tmaps = {}

    for qa, qb in combinations(questionnaire_names, 2):
        pairing_name = f'{qa}_minus_{qb}'
        paired_pmap_loop = (
            outdir
            / f'paired_test_{pairing_name}/onesample_2sided_tfce_corrp_fstat1.nii.gz'
        )
        paired_t_loop = outdir / f'paired_test_{pairing_name}/uncorrected_tstat1.nii.gz'

        question_pmaps[pairing_name] = paired_pmap_loop
        question_tmaps[pairing_name] = paired_t_loop
    return question_pmaps, question_tmaps


def load_mask(mask_path):
    """Load mask and return 1D mask indices and affine/header info."""
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata().astype(bool)
    mask_idx = np.where(mask_data)
    return mask_data, mask_idx, mask_nii.affine, mask_nii.header, mask_nii


def create_binary_matrix(tmap_dict, pmap_dict, mask_idx, p_threshold=0.05):
    """
    Create a binary matrix for positive and negative activations from dictionary of maps.

    Args:
        tmap_dict: dict, key = contrast name, value = tmap path
        pmap_dict: dict, key = contrast name, value = pmap path
        mask_idx: tuple of arrays, mask indices
        threshold: float, threshold for p-value maps (e.g., 0.9 for p<0.1)

    Returns:
        binary_matrix: np.array, shape (n_vox, n_contrasts*2)
        map_keys: list of strings labeling each column with sign-aware contrast
    """
    threshold = 1 - p_threshold
    contrast_names = list(tmap_dict.keys())
    n_vox = len(mask_idx[0])
    n_contrasts = len(contrast_names)

    binary_matrix = np.zeros((n_vox, n_contrasts * 2), dtype=np.uint8)
    map_keys = []

    for i, contrast in enumerate(contrast_names):
        tmap_data = nib.load(tmap_dict[contrast]).get_fdata()[mask_idx]
        pmap_data = nib.load(pmap_dict[contrast]).get_fdata()[mask_idx]

        # Parse A-B from key
        a, b = contrast.split('_minus_')

        # Positive activation
        pos = (tmap_data > 0) & (pmap_data > threshold)
        binary_matrix[:, 2 * i] = pos.astype(np.uint8)
        map_keys.append(f'{b} < {a}')  # swap for positive sign

        # Negative activation
        neg = (tmap_data < 0) & (pmap_data > threshold)
        binary_matrix[:, 2 * i + 1] = neg.astype(np.uint8)
        map_keys.append(f'{a} < {b}')  # keep order for negative sign

        # Clean up
        del tmap_data, pmap_data, pos, neg

    return binary_matrix, map_keys


def integer_conjunction_map(binary_matrix, map_labels=None):
    """
    Convert a binary matrix of voxel activations to integer-encoded conjunctions.

    Args:
        binary_matrix: np.array of shape (n_vox, n_maps), 0/1 values
        map_labels: optional list of length n_maps, names of hypotheses

    Returns:
        integer_map: np.array of shape (n_vox,), each voxel encoded as an integer
        key_map: dict mapping integer -> list of active hypotheses (indices or labels)
    """
    n_vox, n_cols = binary_matrix.shape
    powers = 2 ** np.arange(n_cols, dtype=np.int32)

    # Encode each voxel's active maps as a single integer
    integer_map = binary_matrix.dot(powers)

    # Build key map: integer -> active map indices/labels
    unique_vals = np.unique(integer_map)
    key_map = {}
    for val in unique_vals:
        if val == 0:
            key_map[val] = []
        else:
            # Find which bits are set
            cols = np.where(((val >> np.arange(n_cols)) & 1) == 1)[0]
            if map_labels is not None:
                key_map[val] = [map_labels[i] for i in cols]
            else:
                key_map[val] = cols.tolist()

    return integer_map, key_map


def plot_integer_map_overlay(
    integer_map,
    mask_data,
    mask_idx,
    key_map,
    z_slices,
    mask_nifti,
    omnibus_pmap_file,
    omnibus_threshold=0.05,
    min_cluster_vox=200,
    n_voxel_thresh_plot_save=None,
):
    """
    Plot one figure per conjunction of maps if at least one cluster meets min_cluster_vox.
    Save figure if n_voxels >= n_voxel_thresh_plot_save.
    """
    # Reconstruct 3D integer map
    img_3d = np.zeros(mask_data.shape, dtype=np.int32)
    img_3d[mask_idx] = integer_map

    # Load omnibus F-test map and threshold
    omnibus_img = nib.load(omnibus_pmap_file)
    omnibus_data = omnibus_img.get_fdata()
    omnibus_binary = (omnibus_data > (1 - omnibus_threshold)).astype(np.int32)

    # Define colormap: 0=background, 1=cluster only, 2=omnibus only, 3=overlap
    cmap = ListedColormap(['black', 'deepskyblue', 'yellow', 'forestgreen'])

    for val, cols in key_map.items():
        if val == 0:
            continue  # skip background

        # Step 1: binary mask for this integer value
        cluster_mask = (img_3d == val).astype(np.int32)

        # Step 2: check if any cluster meets min_cluster_vox
        labeled_clusters, n_clusters = ndimage.label(cluster_mask)
        if n_clusters == 0:
            continue
        cluster_sizes = [
            np.sum(labeled_clusters == cl) for cl in range(1, n_clusters + 1)
        ]
        if all(size < min_cluster_vox for size in cluster_sizes):
            continue

        # Step 3: count all nonzero voxels for figure title
        n_voxels = np.sum(cluster_mask > 0)

        # Step 4: create overlay map
        overlay_int = np.zeros(cluster_mask.shape, dtype=np.int32)
        overlay_int[cluster_mask == 1] = 1
        overlay_int[(omnibus_binary == 1) & (cluster_mask == 0)] = 2
        overlay_int[(cluster_mask == 1) & (omnibus_binary == 1)] = 3

        # Step 5: make NIfTI and plot
        overlay_nifti = nib.Nifti1Image(overlay_int, affine=mask_nifti.affine)
        label_text = f'Hypotheses: {", ".join(cols)}'

        display = plot_stat_map(
            overlay_nifti,
            display_mode='z',
            cut_coords=z_slices,
            title=f'{label_text} ({n_voxels} voxels)',
            colorbar=False,
            cmap=cmap,
            symmetric_cbar=False,
        )

        # Save figure if threshold met
        if (n_voxel_thresh_plot_save is not None) and (
            n_voxels >= n_voxel_thresh_plot_save
        ):
            # Construct filename based on cols
            name_raw = '_'.join(cols)
            fig_name = name_raw.replace(' ', '')
            out_path = f'./figures/{fig_name}.png'

            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f'Saved figure: {out_path}')

        plt.show()

        del overlay_int, overlay_nifti, cluster_mask


def plot_integer_map_overlay_OLD(
    integer_map,
    mask_data,
    mask_idx,
    key_map,
    z_slices,
    mask_nifti,
    omnibus_pmap_file,
    omnibus_threshold=0.05,
    min_cluster_vox=200,
):
    """
    Plot one figure per conjunction of maps if at least one cluster meets min_cluster_vox.

    Args:
        integer_map: 1D array of integer-encoded conjunctions
        mask_data, mask_idx: mask info
        key_map: dict mapping integer -> list of map labels
        z_slices: z coordinates for plotting
        mask_nifti: NIfTI object for affine info
        omnibus_pmap_file: NIfTI file for omnibus thresholding
        omnibus_threshold: threshold for omnibus map
        min_cluster_vox: minimum cluster size to trigger plotting
    """
    # Reconstruct 3D integer map
    img_3d = np.zeros(mask_data.shape, dtype=np.int32)
    img_3d[mask_idx] = integer_map

    # Load omnibus F-test map and threshold
    omnibus_img = nib.load(omnibus_pmap_file)
    omnibus_data = omnibus_img.get_fdata()
    omnibus_binary = (omnibus_data > (1 - omnibus_threshold)).astype(np.int32)

    # Define colormap: 0=background, 1=cluster only, 2=omnibus only, 3=overlap
    cmap = ListedColormap(['black', 'blue', 'lemonchiffon', 'limegreen'])

    for val, cols in key_map.items():
        if val == 0:
            continue  # skip background

        # Step 1: binary mask for this integer value
        cluster_mask = (img_3d == val).astype(np.int32)

        # Step 2: check if any cluster meets min_cluster_vox
        labeled_clusters, n_clusters = ndimage.label(cluster_mask)
        if n_clusters == 0:
            continue  # nothing to plot
        cluster_sizes = [
            np.sum(labeled_clusters == cl) for cl in range(1, n_clusters + 1)
        ]
        if all(size < min_cluster_vox for size in cluster_sizes):
            continue  # skip if no cluster is big enough

        # Step 3: count all nonzero voxels for figure title
        n_voxels = np.sum(cluster_mask > 0)

        # Step 4: create overlay map
        overlay_int = np.zeros(cluster_mask.shape, dtype=np.int32)
        overlay_int[cluster_mask == 1] = 1
        overlay_int[(omnibus_binary == 1) & (cluster_mask == 0)] = 2
        overlay_int[(cluster_mask == 1) & (omnibus_binary == 1)] = 3

        # Step 5: make NIfTI and plot
        overlay_nifti = nib.Nifti1Image(overlay_int, affine=mask_nifti.affine)
        label_text = f'Hypotheses: {", ".join(cols)}'
        plot_stat_map(
            overlay_nifti,
            display_mode='z',
            cut_coords=z_slices,
            title=f'{label_text} ({n_voxels} voxels)',
            colorbar=False,
            cmap=cmap,
            symmetric_cbar=False,
        )
        plt.show()
        del overlay_int, overlay_nifti, cluster_mask


# ---------------------------
# Step 1-3: full pipeline
# ---------------------------


def process_pairwise_maps(
    tmap_paths,
    pmap_paths,
    omnibus_pmap_file,
    mask_path,
    z_slices,
    t_threshold=0.05,
    omnibus_f_threshold=0.05,
    min_cluster_vox=200,
    n_voxel_thresh_plot_save=None,
):
    mask_data, mask_idx, affine, header, mask_nii = load_mask(mask_path)

    binary_matrix, map_keys = create_binary_matrix(
        tmap_paths, pmap_paths, mask_idx, t_threshold
    )

    integer_map, integer_keys = integer_conjunction_map(binary_matrix, map_keys)

    plot_integer_map_overlay(
        integer_map,
        mask_data,
        mask_idx,
        integer_keys,
        z_slices,
        mask_nii,
        omnibus_pmap_file,
        omnibus_threshold=omnibus_f_threshold,
        min_cluster_vox=min_cluster_vox,
        n_voxel_thresh_plot_save=n_voxel_thresh_plot_save,
    )
