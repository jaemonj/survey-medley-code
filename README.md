# Survey Medley Data Analysis

This repository contains code for analyzing Survey Medley data, written by Jaemon Jumpawong and Jeanette Mumford, in collaboration with Patrick Bissett. It includes scripts for preprocessing, quality assessment, exploratory analyses, and result generation.

## Quick Start

To quickly browse analysis results:

1. Open the [Master Analysis Registry](analysis_registry/README.md)
2. Click links to the **Results Files** notebooks for each analysis step
3. For deeper exploration or rerunning analyses, navigate the corresponding folder in `analyses/`

---

## Repository Structure

> **Note:** This code is configured for a Sherlock-based directory structure. Paths and file locations are specific to that environment and may need adjustment to run elsewhere.

**Highlights:**

- **analyses/** contains all step-specific scripts and notebooks. Some examples:  
  - `assess_subject_bold_dropout/` – Scripts for quality checks on BOLD data  
  - `f_test_questionnaire_averages/` – Study how questionnaire-based activations differ  

- **analysis_registry/** contains the Master Analysis Registry, designed for internal navigation and quick reference — it makes it easy to locate results, notebooks, and processed outputs without manually searching directories.  
  - `README.md`: [Master Analysis Registry](analysis_registry/README.md)  
    - Quickly summarizes results  
    - Helps navigate the analysis code  
    - Points to output files  
  - Each analysis step is documented in a YAML file (`analysis_registry/analyses/*.yaml`)  
  - `build_master_registry.py`: Assembles individual `.yaml` files into the main `.yaml` and `README.md`

- **src/** contains utility code and core processing functions used across analyses.

---

## Running the Code

- **Python 3.12+** is recommended.  
- All scripts assume paths and files are structured according to the Sherlock environment.  
- Most analyses are run via batch scripts on Sherlock; notebooks are primarily for exploration, visualization, and QA.

---

## License & Citation

- MIT License  
- If this code is used for research purposes, please cite the corresponding study or repository.
