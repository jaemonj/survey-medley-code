#!/bin/bash
##
#SBATCH -J randomise_questionnaire_brain_behavior_slope_ftest
#SBATCH --time=36:00:00
#SBATCH --begin=now
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH -p russpold,hns,normal
#SBATCH -o log/%x-%A-%a.out
#SBATCH -e log/%x-%A-%a.err
#SBATCH --mail-user=jaemon@stanford.edu
#SBATCH --mail-type=FAIL
# ------------------------------------------

. ${HOME}/.bashrc  
micromamba activate fmri_analysis
module load contribs 
module load poldrack 
module load fsl 

bash /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/questionnaire_brain_behavior_slope_ftest/output_glm_ftest/randomise_call.sh
