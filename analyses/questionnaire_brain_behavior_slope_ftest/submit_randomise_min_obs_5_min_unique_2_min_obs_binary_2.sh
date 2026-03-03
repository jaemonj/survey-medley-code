#!/bin/bash
##
#SBATCH -J brain_behavior_slope_ftest_min_obs_5_min_unique_2_min_obs_binary_2
#SBATCH --time=36:00:00
#SBATCH --begin=now
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p russpold,hns,normal
#SBATCH --output=/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/survey_medley_code/log/%x-%A-%a.out
#SBATCH --error=/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/survey_medley_code/log/%x-%A-%a.err
#SBATCH --mail-user=jaemon@stanford.edu
#SBATCH --mail-type=FAIL
# ------------------------------------------

module load contribs 
module load poldrack

PROJECT_ROOT="/oak/stanford/groups/russpold/data/uh2/aim1/analysis_code/survey_medley_code"
UV_SETUP="$PROJECT_ROOT/setup_uv_sherlock.sh"

source $UV_SETUP

/oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/questionnaire_brain_behavior_slope_ftest/output_glm_ftest_min_obs_5_min_unique_2_min_obs_binary_2/randomise_call.sh
