from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting
from nilearn.image import load_img
from nilearn.masking import intersect_masks
from scipy.stats import entropy


def create_group_mask(sub_ids, fmri_prep_dir):
    masks = []
    for sub_id in sub_ids:
        mask_files = glob(
            f'{fmri_prep_dir}/sub-s{sub_id}/ses*/func/*surveyMedley*mask.nii.gz'
        )
        if not mask_files:
            print(f'Warning: No mask found for subject {sub_id}')
            continue
        if len(mask_files) > 1:
            print(
                f'Warning: Multiple masks found for subject {sub_id}, using the first one'
            )
        mask_path = mask_files[0]
        masks.append(load_img(mask_path))

    if not masks:
        raise ValueError('No masks were found for any subjects')

    # Create a mask that includes voxels present in 100% of subject masks
    group_mask = intersect_masks(masks, threshold=1.0)

    return group_mask


def all_partitions(n_levels):
    """
    Generate all partitions of n_levels elements, excluding the all-in-one and all-separate partitions.
    """
    if not isinstance(n_levels, int) or n_levels <= 0:
        raise ValueError('n_levels must be a positive integer')

    def partitions(set_):
        if not set_:
            yield []
            return
        first = set_[0]
        for smaller in partitions(set_[1:]):
            for i, subset in enumerate(smaller):
                yield smaller[:i] + [[first] + subset] + smaller[i + 1 :]
            yield [[first]] + smaller

    all_parts = list(partitions(list(range(1, n_levels + 1))))  # Start from 1

    # Remove the all-in-one partition and the all-separate partition
    return [p for p in all_parts if len(p) != 1 and len(p) != n_levels]


def center_within_subject(brain_data, sub_id):
    """
    Center brain data within each subject.

    Parameters:
    -----------
    brain_data : numpy.ndarray
        Array of shape (n_measures, n_vox) containing brain activation data.
    sub_id : numpy.ndarray
        Array of length n_measures with subject identifiers.

    Returns:
    --------
    numpy.ndarray
        Centered brain_data (within-subject, per voxel).

    Raises:
    -------
    ValueError
        If brain_data first dimension doesn't match length of sub_id,
        or if no data is found for a subject.
    """
    if brain_data.shape[0] != len(sub_id):
        raise ValueError('brain_data first dimension must match length of sub_id')

    centered_data = brain_data.copy()
    for subj in np.unique(sub_id):
        idx = np.where(sub_id == subj)[0]
        if len(idx) == 0:
            raise ValueError(f'No data found for subject {subj}')
        centered_data[idx, :] -= centered_data[idx, :].mean(axis=0, keepdims=True)
    return centered_data


def partition_score_all_voxels(brain_data, sub_id, item_id, partition):
    """
    Calculate the partition score (extended Calinski-Harabasz score) for all voxels.

    Parameters:
    -----------
    brain_data : numpy.ndarray
        Array of shape (n_measures, n_vox) containing brain activation data.
    sub_id : numpy.ndarray
        Array of length n_measures with subject identifiers.
    item_id : numpy.ndarray
        Array of length n_measures with item identifiers.
    partition : list
        List of lists representing the partition of items.

    Returns:
    --------
    numpy.ndarray
        Array of Calinski-Harabasz scores for each voxel.

    Raises:
    -------
    ValueError
        If inputs have inconsistent shapes or if data cannot be reshaped as expected.
    """
    if brain_data.shape[0] != len(sub_id) or brain_data.shape[0] != len(item_id):
        raise ValueError(
            'brain_data first dimension must match length of sub_id and item_id'
        )

    n_voxels = brain_data.shape[1]
    n_subjects = len(np.unique(sub_id))
    n_items = len(np.unique(item_id))

    if n_subjects * n_items != brain_data.shape[0]:
        raise ValueError(
            'Mismatch between number of subjects, items, and brain_data shape'
        )

    try:
        brain_data_by_subject = brain_data.reshape(n_subjects, -1, n_voxels)
    except ValueError:
        raise ValueError(
            'Unable to reshape brain_data. Check if n_subjects and n_items are correct.'
        )

    subject_means = np.mean(brain_data_by_subject, axis=1)
    c = np.mean(subject_means, axis=0)

    bcss = np.zeros(n_voxels)
    wcss = np.zeros(n_voxels)

    for cluster in partition:
        cluster_mask = np.isin(item_id.reshape(n_subjects, -1), cluster)
        print(f'Cluster: {cluster}, Mask sum: {np.sum(cluster_mask)}')
        if np.sum(cluster_mask) == 0:
            print(f'Warning: Empty cluster {cluster}')
            continue

        cluster_data = brain_data_by_subject[cluster_mask].reshape(
            n_subjects, -1, n_voxels
        )

        cluster_subject_means = np.mean(cluster_data, axis=1)
        ci = np.mean(cluster_subject_means, axis=0)

        ni = len(cluster)

        bcss += ni * (ci - c) ** 2

        item_means = np.mean(cluster_data, axis=0)
        wcss += np.sum((item_means - ci) ** 2, axis=0)

    k = len(partition)
    ch_score = (bcss / (k - 1)) / (wcss / (n_items - k))
    epsilon = 1e-10
    ch_score = (bcss / (k - 1)) / (wcss / (n_items - k) + epsilon)
    return ch_score


def run_partition_analysis(brain_data, sub_id, item_id, n_levels, brain_masker):
    """
    Run partition analysis on brain data.

    Parameters:
    -----------
    brain_data : numpy.ndarray
        Array of shape (n_measures, n_vox) containing brain activation data.
    sub_id : numpy.ndarray
        Array of length n_measures with subject identifiers.
    item_id : numpy.ndarray
        Array of length n_measures with item identifiers.
    n_levels : int
        Number of distinct items.
    brain_masker : nilearn.input_data.NiftiMasker
        Fitted brain masker for transforming between voxel space and image space.

    Returns:
    --------
    tuple
        Contains:
        - best_partition_map : numpy.ndarray
        - max_partition_score_map : numpy.ndarray
        - score_gap_map : numpy.ndarray
        - entropy_map : numpy.ndarray
        - partition_key : dict

    Raises:
    -------
    ValueError
        If inputs have inconsistent shapes or if n_levels doesn't match unique item_id count.
    """
    if brain_data.shape[0] != len(sub_id) or brain_data.shape[0] != len(item_id):
        raise ValueError(
            'brain_data first dimension must match length of sub_id and item_id'
        )

    if len(np.unique(item_id)) != n_levels:
        raise ValueError('Number of unique item_id values must match n_levels')

    n_vox = brain_data.shape[1]

    partitions = all_partitions(n_levels)
    partition_key = {i + 1: p for i, p in enumerate(partitions)}

    best_partition_map = np.zeros(n_vox, dtype=int)
    max_partition_score_map = np.zeros(n_vox)
    score_gap_map = np.zeros(n_vox)
    entropy_map = np.zeros(n_vox)

    all_scores = np.zeros((len(partitions), n_vox))

    for i, partition in enumerate(partitions):
        scores = partition_score_all_voxels(brain_data, sub_id, item_id, partition)
        all_scores[i] = scores
        print(
            f'Partition {partition}: Min score = {np.min(scores)}, Max score = {np.max(scores)}'
        )

    best_partition_map = np.argmax(all_scores, axis=0) + 1
    max_partition_score_map = np.max(all_scores, axis=0)
    unique_best_partitions = np.unique(best_partition_map)
    print(f'Unique best partitions: {unique_best_partitions}')
    for partition_id in unique_best_partitions:
        count = np.sum(best_partition_map == partition_id)
        print(f'Partition {partition_key[partition_id]} selected for {count} voxels')

    sorted_scores = np.sort(all_scores, axis=0)
    score_gap_map = 1 - (sorted_scores[-2] / sorted_scores[-1])

    norm_scores = all_scores / np.sum(all_scores, axis=0)
    entropy_map = entropy(norm_scores, axis=0)

    return (
        best_partition_map,
        max_partition_score_map,
        score_gap_map,
        entropy_map,
        partition_key,
    )


def make_brain_maps(
    best_partition_map,
    max_partition_score_map,
    score_gap_map,
    entropy_map,
    brain_masker,
):
    """
    Save the computed maps as NIfTI images.

    Parameters:
    -----------
    best_partition_map : numpy.ndarray
        Array of best partition IDs for each voxel.
    max_partition_score_map : numpy.ndarray
        Array of maximum partition scores for each voxel.
    score_gap_map : numpy.ndarray
        Array of score gaps for each voxel.
    entropy_map : numpy.ndarray
        Array of entropy values for each voxel.
    brain_masker : nilearn.input_data.NiftiMasker
        Fitted brain masker for transforming between voxel space and image space.
    out_prefix : str
        Prefix for output filenames.

    Returns:
    --------
    tuple
        Contains the NIfTI image objects for each map.

    Raises:
    -------
    ValueError
        If input maps have inconsistent shapes or if there's an error in inverse transform.
    """
    if not all(
        map.shape == best_partition_map.shape
        for map in [max_partition_score_map, score_gap_map, entropy_map]
    ):
        raise ValueError('All input maps must have the same shape')

    try:
        best_img = brain_masker.inverse_transform(best_partition_map)
        max_score_img = brain_masker.inverse_transform(max_partition_score_map)
        gap_img = brain_masker.inverse_transform(score_gap_map)
        entropy_img = brain_masker.inverse_transform(entropy_map)
    except Exception as e:
        raise ValueError(f'Error in inverse transform: {str(e)}')
    return best_img, max_score_img, gap_img, entropy_img


def plot_maps(maps, map_titles, slices, out_pdf, savefig=False):
    """
    Plot NIfTI maps using nilearn.

    Parameters:
    -----------
    maps : list
        List of NIfTI image objects to plot.
    map_titles : list
        List of titles for each map.
    slices : list
        List of z-slice coordinates to display.
    out_pdf : str
        Output filename for the PDF.
    savefig : bool, optional
        Whether to save the figure to a file (default is False).

    Raises:
    -------
    ValueError
        If maps and map_titles have different lengths or if maps are not NIfTI image objects.
    """
    if len(maps) != len(map_titles):
        raise ValueError(
            f'maps (n={len(maps)}) and map_titles (n={len(map_titles)}) must have the same length'
        )

    if not all(isinstance(img, Nifti1Image) for img in maps):
        raise ValueError('All maps must be NIfTI image objects')

    n_rows = len(maps)
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for ax, img, title in zip(axes, maps, map_titles):
        plotting.plot_stat_map(
            img,
            display_mode='z',
            cut_coords=slices,
            colorbar=True,
            axes=ax,
            cmap='viridis',
            title=title,
            draw_cross=False,
        )

    plt.tight_layout()

    if savefig:
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_partition_grid(
    partition_key, n_levels, label_mapper=None, out_file='partition_grid.pdf'
):
    """
    Plot a grid visualization of partitions.

    Parameters:
    -----------
    partition_key : dict
        Dictionary of partitions, e.g., {1: [[1,2],[3,4,5]], 2: [[1],[2,3,4,5]], ...}
    n_levels : int
        Number of questionnaire items.
    label_mapper : dict, optional
        Dictionary mapping index to item names.
    out_file : str, optional
        Output filename for the PDF.

    Raises:
    -------
    ValueError
        If inputs have incorrect types or if n_levels is not a positive integer.
    """
    if not isinstance(partition_key, dict):
        raise ValueError('partition_key must be a dictionary')

    if not isinstance(n_levels, int) or n_levels <= 0:
        raise ValueError('n_levels must be a positive integer')

    if label_mapper is not None and not isinstance(label_mapper, dict):
        raise ValueError('label_mapper must be a dictionary or None')

    n_partitions = len(partition_key)

    grid = np.zeros((n_levels, n_partitions), dtype=int)
    for pid, clusters in partition_key.items():
        for cluster_id, cluster in enumerate(clusters, start=1):
            for idx in cluster:
                grid[idx - 1, pid - 1] = (
                    cluster_id  # Subtract 1 from idx for 0-based array indexing
                )

    plt.figure(
        figsize=(max(6, n_partitions / 5), 4)
    )  # Adjust figure size based on number of partitions
    cmap = plt.get_cmap('tab20', np.max(grid))
    plt.imshow(grid, cmap=cmap, aspect='auto')

    plt.xticks(
        ticks=np.arange(n_partitions),
        labels=np.arange(1, n_partitions + 1),
        rotation=90,
        fontsize=8,
    )

    if label_mapper:
        plt.yticks(
            ticks=np.arange(n_levels),
            labels=[label_mapper.get(i + 1, f'Q{i + 1}') for i in range(n_levels)],
            fontsize=8,
        )
    else:
        plt.yticks(
            ticks=np.arange(n_levels),
            labels=[f'Q{i + 1}' for i in range(n_levels)],
            fontsize=8,
        )

    plt.xlabel('Partition ID')
    plt.ylabel('Questionnaire Summary')
    plt.title('Partition Grid (columns = partitions, rows = items, colors = cluster)')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# ----------------------
# Searchlight functions
# ----------------------

from scipy.spatial.distance import cdist


def get_voxel_coordinates(mask_img):
    """Get the 3D coordinates of voxels in the mask."""
    mask_data = mask_img.get_fdata()
    return np.array(np.where(mask_data)).T


def get_searchlight_indices(voxel_coords, center_voxel_idx, radius):
    """Get indices of voxels within the searchlight radius."""
    center_coord = voxel_coords[center_voxel_idx]
    distances = cdist([center_coord], voxel_coords)[0]
    return np.where(distances <= radius)[0]


import time


def partition_score_searchlight(searchlight_data, sub_id, item_id, partition):
    n_subjects = len(np.unique(sub_id))
    n_items = len(np.unique(item_id))

    sort_idx = np.lexsort((item_id, sub_id))
    data_by_subject = searchlight_data[sort_idx].reshape(n_subjects, n_items, -1)

    overall_centroid = np.mean(data_by_subject, axis=(0, 1))

    bcss = 0.0
    wcss = 0.0

    for cluster in partition:
        cluster_mask = np.isin(np.arange(n_items), [i - 1 for i in cluster])
        cluster_data = data_by_subject[:, cluster_mask, :]

        cluster_centroid = np.mean(cluster_data, axis=(0, 1))

        n_points_cluster = n_subjects * len(cluster)
        bcss += n_points_cluster * np.sum((cluster_centroid - overall_centroid) ** 2)

        diffs = cluster_data - cluster_centroid
        wcss += np.sum(diffs**2)

    k = len(partition)

    if k <= 1 or n_subjects * n_items <= k:
        return 0.0

    ch_score = (bcss / (k - 1)) / (wcss / (n_subjects * n_items - k) + 1e-10)

    return 0.0 if np.isnan(ch_score) or np.isinf(ch_score) else ch_score


def precompute_partition_masks(n_items, partitions):
    masks = []
    for partition in partitions:
        mask = np.zeros(n_items, dtype=int)
        for cluster_idx, cluster in enumerate(partition):
            for item in cluster:
                mask[item - 1] = cluster_idx + 1  # Use cluster_idx + 1 to avoid 0
        masks.append(mask)
    return np.array(masks)


def searchlight_partition_analysis(
    brain_data, voxel_coords, sub_id, item_id, radius=3, n_voxels=None
):
    """Run searchlight partition analysis on 2D brain data."""
    total_voxels = brain_data.shape[1]
    n_voxels = n_voxels or total_voxels
    n_subjects = len(np.unique(sub_id))
    n_items = len(np.unique(item_id))

    print(f'Number of subjects: {n_subjects}')
    print(f'Number of items: {n_items}')
    print(f'Shape of brain_data: {brain_data.shape}')

    partitions = all_partitions(n_items)
    print(f'Number of partitions: {len(partitions)}')
    print('First few partitions:')
    for i, p in enumerate(partitions[:5]):
        print(f'  {i + 1}: {p}')

    partition_key = {i + 1: p for i, p in enumerate(partitions)}

    # Initialize maps for all voxels
    best_partition_map = np.zeros(total_voxels, dtype=int)
    max_partition_score_map = np.zeros(total_voxels)
    score_gap_map = np.zeros(total_voxels)
    entropy_map = np.zeros(total_voxels)

    start_time = time.time()

    # Precompute searchlight indices for all voxels
    all_searchlight_indices = [
        get_searchlight_indices(voxel_coords, voxel_idx, radius)
        for voxel_idx in range(total_voxels)
    ]

    # Create a random permutation of voxel indices if n_voxels is specified
    if n_voxels and n_voxels < total_voxels:
        voxel_indices = np.random.permutation(total_voxels)[:n_voxels]
    else:
        voxel_indices = range(total_voxels)

    for i, voxel_idx in enumerate(voxel_indices):
        if (i + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(
                f'Processed {i + 1} out of {n_voxels} voxels. Elapsed time: {elapsed_time:.2f} seconds'
            )

        searchlight_indices = all_searchlight_indices[voxel_idx]
        searchlight_data = brain_data[:, searchlight_indices]

        all_scores = np.array(
            [
                partition_score_searchlight(
                    searchlight_data, sub_id, item_id, partition
                )
                for partition in partitions
            ]
        )

        best_partition = np.argmax(all_scores) + 1
        max_score = np.max(all_scores)

        sorted_scores = np.sort(all_scores)
        score_gap = 1 - (sorted_scores[-2] / (sorted_scores[-1] + 1e-10))

        norm_scores = all_scores / (np.sum(all_scores) + 1e-10)
        entropy_value = entropy(norm_scores)

        best_partition_map[voxel_idx] = best_partition
        max_partition_score_map[voxel_idx] = max_score
        score_gap_map[voxel_idx] = score_gap
        entropy_map[voxel_idx] = entropy_value

    total_time = time.time() - start_time
    print(
        f'Finished processing {n_voxels} voxels. Total time: {total_time:.2f} seconds'
    )

    return (
        best_partition_map,
        max_partition_score_map,
        score_gap_map,
        entropy_map,
        partition_key,
    )


# =======================================
# new code

import os

from joblib import Parallel, delayed
from scipy.spatial import cKDTree

# ----------------------
# Helper / precompute
# ----------------------


def precompute_partition_masks_list(n_items, partitions):
    """
    For each partition (a list of clusters where items are 1-based),
    return a list of clusters, each cluster being a zero-based item index array.
    Example: partitions = [[ [1,2], [3] ], [[1],[2,3]] ]
    returns: [ [array([0,1]), array([2])], [array([0]), array([1,2])] ]
    """
    partitions_cluster_indices = []
    for partition in partitions:
        clusters = [
            np.array([i - 1 for i in cluster], dtype=int) for cluster in partition
        ]
        partitions_cluster_indices.append(clusters)
    return partitions_cluster_indices


def compute_CH_for_partitions(data_by_subject, partitions_cluster_indices):
    """
    data_by_subject: shape (n_subjects, n_items, n_features)
    partitions_cluster_indices: list of partitions; each partition is list of cluster-item-index arrays
    returns: 1D numpy array of CH scores (len = number of partitions)
    """
    n_subjects, n_items, n_features = data_by_subject.shape
    overall_centroid = data_by_subject.mean(axis=(0, 1))  # shape (n_features,)

    n_points_total = n_subjects * n_items

    scores = np.empty(len(partitions_cluster_indices), dtype=float)

    # compute for each partition
    for p_idx, clusters in enumerate(partitions_cluster_indices):
        bcss = 0.0
        wcss = 0.0
        k = len(clusters)
        # skip degenerate
        if k <= 1 or n_points_total <= k:
            scores[p_idx] = 0.0
            continue

        for cluster_indices in clusters:
            # cluster_data shape: (n_subjects, n_cluster_items, n_features)
            cluster_data = data_by_subject[:, cluster_indices, :]
            # centroid over subjects and items
            cluster_centroid = cluster_data.mean(axis=(0, 1))
            n_points_cluster = n_subjects * cluster_indices.size
            bcss += n_points_cluster * np.sum(
                (cluster_centroid - overall_centroid) ** 2
            )
            # within-cluster sum of squares
            diffs = cluster_data - cluster_centroid  # broadcast
            wcss += np.sum(diffs * diffs)

        # CH index
        denom = (wcss / (n_points_total - k)) + 1e-10
        ch = (bcss / (k - 1)) / denom
        if not np.isfinite(ch):
            ch = 0.0
        scores[p_idx] = ch

    return scores


# ----------------------
# Main searchlight (parallel, KDTree)
# ----------------------


def searchlight_partition_analysis_fast(
    brain_data,  # shape (n_observations, n_voxels)
    voxel_coords,  # shape (n_voxels, 3)
    sub_id,  # length n_observations
    item_id,  # length n_observations
    radius=3.0,
    n_voxels_to_run=None,
    n_jobs=None,
    verbose=True,
):
    """
    Faster searchlight partition analysis.
    - This function returns the same tuple as before:
      (best_partition_map, max_partition_score_map, score_gap_map, entropy_map, partition_key)
    """

    # basic sizing
    total_voxels = brain_data.shape[1]
    n_jobs = n_jobs or min(8, max(1, (os.cpu_count() or 1) - 1))
    n_subjects = len(np.unique(sub_id))
    n_items = len(np.unique(item_id))

    # Precompute partitions and helper structures
    partitions = all_partitions(n_items)
    partitions_cluster_indices = precompute_partition_masks_list(n_items, partitions)
    partition_key = {i + 1: p for i, p in enumerate(partitions)}

    # Precompute sort index and sorted brain_data (so we don't sort per voxel)
    sort_idx = np.lexsort((item_id, sub_id))
    brain_data_sorted = brain_data[sort_idx, :]  # shape (n_observations, n_voxels)
    # we will also need the mapping to reshape into (n_subjects, n_items, n_features)

    # KDTree for neighbor queries (fast)
    tree = cKDTree(voxel_coords)

    # choose voxels to run
    if n_voxels_to_run and n_voxels_to_run < total_voxels:
        voxel_indices = np.random.permutation(total_voxels)[:n_voxels_to_run]
    else:
        voxel_indices = np.arange(total_voxels)

    # outputs
    best_partition_map = np.zeros(total_voxels, dtype=int)
    max_partition_score_map = np.zeros(total_voxels, dtype=float)
    score_gap_map = np.zeros(total_voxels, dtype=float)
    entropy_map = np.zeros(total_voxels, dtype=float)

    if verbose:
        print(f'Number of subjects: {n_subjects}')
        print(f'Number of items: {n_items}')
        print(f'Number of partitions: {len(partitions)}')
        print(f'Using n_jobs={n_jobs}, radius={radius}')
        start_time = time.time()

    # define voxel worker
    def process_voxel(voxel_idx_local, voxel_global_idx):
        # find neighbors using KDTree (includes the center itself if within points)
        neigh_idxs = tree.query_ball_point(voxel_coords[voxel_global_idx], r=radius)
        if len(neigh_idxs) == 0:
            # empty searchlight -> set zeros
            return (voxel_global_idx, 0, 0.0, 1.0, 0.0)

        # extract searchlight data (rows already sorted)
        # shape (n_observations, n_features)
        searchlight_data = brain_data_sorted[:, neigh_idxs]

        # reshape into (n_subjects, n_items, n_features)
        try:
            data_by_subject = searchlight_data.reshape(n_subjects, n_items, -1)
        except Exception:
            # in case the sort / counts mismatch, fallback to re-sorting by idx
            # but this should not happen if inputs are consistent
            raise

        # compute CH scores for all partitions
        all_scores = compute_CH_for_partitions(
            data_by_subject, partitions_cluster_indices
        )

        # outputs derived
        best_partition_idx = int(np.argmax(all_scores)) + 1
        max_score = float(np.max(all_scores))
        # compute score gap safely
        sorted_scores = np.sort(all_scores)
        if sorted_scores[-1] <= 0:
            score_gap = 0.0
        else:
            second = sorted_scores[-2] if sorted_scores.size >= 2 else 0.0
            score_gap = float(1.0 - (second / (sorted_scores[-1] + 1e-10)))
        # entropy
        norm_scores = all_scores / (np.sum(all_scores) + 1e-10)
        ent = float(entropy(norm_scores))

        return (voxel_global_idx, best_partition_idx, max_score, score_gap, ent)

    # Parallel loop
    # We use threads because numpy's heavy ops release the GIL; this avoids copying large brain_data to subprocesses.
    # If you prefer process-based parallelism and have memory to spare, set prefer="processes" and backend="loky"
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(process_voxel)(i, int(vidx)) for i, vidx in enumerate(voxel_indices)
    )

    # write results back into maps
    for voxel_global_idx, best_p, max_s, gap, ent in results:
        best_partition_map[voxel_global_idx] = best_p
        max_partition_score_map[voxel_global_idx] = max_s
        score_gap_map[voxel_global_idx] = gap
        entropy_map[voxel_global_idx] = ent

    if verbose:
        total_time = time.time() - start_time
        print(
            f'Finished processing {len(voxel_indices)} voxels. Total time: {total_time:.2f} s'
        )

    return (
        best_partition_map,
        max_partition_score_map,
        score_gap_map,
        entropy_map,
        partition_key,
    )
