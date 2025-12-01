import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from nilearn import datasets, plotting
from nilearn.image import load_img
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------- Data Utilities ---------------------- #


def load_nifti_data(nifti_paths: List[str]) -> np.ndarray:
    """Load multiple NIfTI files into a 4D array (subjects x voxels)."""
    data_list = [load_img(p).get_fdata() for p in nifti_paths]
    return np.array(data_list, dtype=np.float32)


def get_symmetric_percentile_bounds(
    nifti_paths: List[str], percentile: float = 98
) -> float:
    """Compute symmetric percentile bounds for multiple NIfTI files."""
    data = load_nifti_data(nifti_paths)
    flat_data = np.concatenate([d.ravel() for d in data])
    flat_data = flat_data[np.isfinite(flat_data)]
    if flat_data.size == 0:
        raise ValueError('No valid data found in NIfTI files.')
    return np.percentile(np.abs(flat_data), percentile)


def get_outlier_voxel_percentages(
    nifti_paths: List[str], n_std: float = 2
) -> List[float]:
    """Compute percentage of outlier voxels per subject."""
    data = load_nifti_data(nifti_paths)
    mean_vol = np.mean(data, axis=0)
    std_vol = np.std(data, axis=0)
    valid_mask = np.isfinite(std_vol) & (std_vol > 1e-6)

    lower = mean_vol - n_std * std_vol
    upper = mean_vol + n_std * std_vol

    percentages = [
        100
        * np.sum(((subject < lower) | (subject > upper)) & valid_mask)
        / np.sum(valid_mask)
        for subject in data
    ]
    return percentages


def get_mean_std_bounds(
    nifti_paths: List[str], n_std: float = 2
) -> Tuple[float, float]:
    """Compute mean ± n_std * std bounds for multiple NIfTI files."""
    data = load_nifti_data(nifti_paths)
    flat_data = np.concatenate([d.ravel() for d in data])
    flat_data = flat_data[np.isfinite(flat_data)]
    mean, std = np.mean(flat_data), np.std(flat_data)
    return mean - n_std * std, mean + n_std * std


# ---------------------- Plotting Utilities ---------------------- #


def _plot_combined_histogram(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['image_outlier_percentage'], bins=30, kde=False, ax=ax)
    ax.set(
        title='Distribution of Outlier Percentages\n(All Input Files)',
        xlabel='Input Image Outlier Percentage',
        ylabel='Frequency',
    )
    path = output_dir / 'outlier_percentage_dist_all.png'
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_faceted_histogram(df: pd.DataFrame, output_dir: Path) -> Path:
    g = sns.displot(
        df,
        x='image_outlier_percentage',
        col='contrast_name',
        col_wrap=5,
        bins=20,
        facet_kws={'sharex': False, 'sharey': False},
        height=3,
        aspect=1.2,
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('Image-Specific Outlier Percentage', 'Frequency')
    path = output_dir / 'outlier_percentage_dist_by_image.png'
    plt.tight_layout()
    g.savefig(path, dpi=300)
    plt.close(g.fig)
    return path


def summarize_outlier_percentages(
    df_list: List[pd.DataFrame], output_dir: Path, temp_dir: Optional[Path] = None
) -> List[Path]:
    temp_dir = temp_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df = pd.concat(df_list, ignore_index=True)
    all_hist = _plot_combined_histogram(combined_df, temp_dir)
    faceted_hist = _plot_faceted_histogram(combined_df, temp_dir)
    (output_dir / 'percent_outlier_data.csv').write_text(
        combined_df.to_csv(index=False)
    )
    return [all_hist, faceted_hist]


def _plot_subject_grid(
    subject_labels: List[str],
    nifti_paths: List[str],
    outlier_percentages: List[float],
    mni_mask: np.ndarray,
    contrast_name: str,
    vmax: float,
    vmin: float,
    colorbar_title: Optional[str],
    n_std: float,
) -> plt.Figure:
    n_subjects = len(subject_labels)
    ncols = (
        4
        if n_subjects <= 12
        else 5
        if n_subjects <= 30
        else 6
        if n_subjects <= 60
        else 10
    )
    nrows = int(np.ceil(n_subjects / ncols))

    fig = plt.figure(figsize=(ncols * 2.0, nrows * 1.6 + 1.5))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.25)
    title_fontsize = 9 if n_subjects <= 20 else 7 if n_subjects <= 50 else 5

    for i, (label, path, outlier) in enumerate(
        zip(subject_labels, nifti_paths, outlier_percentages)
    ):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        display = plotting.plot_stat_map(
            path,
            display_mode='z',
            cut_coords=[5],
            colorbar=False,
            vmax=vmax,
            vmin=vmin,
            axes=ax,
            bg_img=None,
            annotate=False,
        )
        display.add_contours(mni_mask, colors='greenyellow', linewidths=1.5)
        ax.set_title(
            f'{label}\n({outlier:.1f}% > {n_std}SD)', fontsize=title_fontsize, pad=4
        )

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    ColorbarBase(
        cbar_ax, cmap=get_cmap('cold_hot'), norm=Normalize(vmin=vmin, vmax=vmax)
    ).set_label(colorbar_title, fontsize=10)
    fig.suptitle(contrast_name, fontsize=14, y=0.98)
    return fig


# ---------------------- Data Structures ---------------------- #


@dataclass
class DataDictionary:
    main_title: str
    nifti_paths: List[str]
    image_labels: List[str]
    data_type_label: str


def validate_data_dictionary(data_dict: Dict[str, Any]) -> DataDictionary:
    required_keys = ['main_title', 'nifti_paths', 'image_labels', 'data_type_label']
    for k in required_keys:
        if k not in data_dict:
            raise ValueError(f'Missing required key: {k}')
    if len(data_dict['nifti_paths']) != len(data_dict['image_labels']):
        raise ValueError('nifti_paths and image_labels must have same length')
    return DataDictionary(**{k: data_dict[k] for k in required_keys})


# ---------------------- PNG / PDF Generation ---------------------- #


def generate_png_files_sd_from_mean(
    image_labels: List[str],
    nifti_paths: List[str],
    output_dir: Path,
    main_title: str,
    colorbar_title: Optional[str] = None,
    n_std: float = 2,
) -> Tuple[Path, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_percentages = get_outlier_voxel_percentages(nifti_paths, n_std)
    vmax = get_symmetric_percentile_bounds(nifti_paths)
    vmin = -vmax
    mni_mask = datasets.load_mni152_brain_mask()
    fig = _plot_subject_grid(
        image_labels,
        nifti_paths,
        outlier_percentages,
        mni_mask,
        main_title,
        vmax,
        vmin,
        colorbar_title,
        n_std,
    )
    png_path = output_dir / f'{main_title}_slice_grid.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    del fig
    gc.collect()
    time.sleep(0.5)
    df_summary = pd.DataFrame(
        {
            'subject_label': image_labels,
            'image_outlier_percentage': outlier_percentages,
            'contrast_name': [main_title] * len(image_labels),
        }
    )
    return png_path, df_summary


def combine_pngs_to_pdf(png_files: List[Path], pdf_path: Path) -> None:
    if not png_files:
        print('No PNGs to combine')
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(f).convert('RGB') for f in png_files]
    images[0].save(pdf_path, save_all=True, append_images=images[1:])
    print(f'PDF saved to {pdf_path}')


# ---------------------- High-level Summary ---------------------- #


def generate_all_data_summaries(
    data_dicts: List[Dict[str, Any]],
    n_std: float = 2,
    output_dir: Path = Path('./data_summary_output'),
) -> None:
    temp_dir = output_dir / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    validated_data = [validate_data_dictionary(d) for d in data_dicts]

    png_paths, all_dfs = [], []
    for data in validated_data:
        print(f'Processing {data.main_title}')
        png_path, df = generate_png_files_sd_from_mean(
            data.image_labels,
            data.nifti_paths,
            temp_dir,
            data.main_title,
            colorbar_title=data.data_type_label,
            n_std=n_std,
        )
        png_paths.append(png_path)
        all_dfs.append(df)

    hist_paths = summarize_outlier_percentages(all_dfs, output_dir, temp_dir)
    combine_pngs_to_pdf(hist_paths + png_paths, output_dir / 'outlier_analysis.pdf')
    shutil.rmtree(temp_dir)


# parallel version

def _process_single_contrast(data: Dict[str, Any], temp_dir: Path, n_std: float):
    validated = validate_data_dictionary(data)
    contrast_temp_dir = temp_dir / validated.main_title
    contrast_temp_dir.mkdir(parents=True, exist_ok=True)

    png_path, df_summary = generate_png_files_sd_from_mean(
        validated.image_labels,
        validated.nifti_paths,
        contrast_temp_dir,
        validated.main_title,
        colorbar_title=validated.data_type_label,
        n_std=n_std,
    )
    return {'png_path': png_path, 'df_summary': df_summary, 'title': validated.main_title}


def generate_all_data_summaries_parallel(
    data_dicts: List[Dict[str, Any]],
    n_std: float = 2,
    output_dir: Path = Path('./data_summary_output'),
    n_workers: int = 2,
):
    temp_dir = output_dir / 'temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    png_paths, all_dfs = [], []

    # Run each contrast in a separate process
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_contrast, d, temp_dir, n_std): d for d in data_dicts}

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f'Finished {result["title"]}')
                png_paths.append(result['png_path'])
                all_dfs.append(result['df_summary'])
            except Exception as e:
                data = futures[future]
                print(f'Error processing {data.get("main_title", "unknown")}: {e}')

    # Summarize outliers (histograms + CSV)
    hist_paths = summarize_outlier_percentages(all_dfs, output_dir, temp_dir)

    # Combine everything into PDF
    combine_pngs_to_pdf(hist_paths + png_paths, output_dir / 'outlier_analysis.pdf')

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f'All done! PDF saved to {output_dir / "outlier_analysis.pdf"}')