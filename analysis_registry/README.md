# Master Analysis Registry

## Summary

<table>
<tr><th colspan="5" style="background-color:#f2f2f2; font-size:1.1em;">Preprocessing</th></tr>
<tr><th>ID</th><th>Description</th><th>Status</th><th>Results Files</th><th>Notes</th></tr>
<tr><td>assess subject bold dropout</td><td>The overlap of the individual masker masks with a generous aggregate mask (voxels with data for at least 30% of the subjects) is used to identify subjects with poor coverage that may be considered for  removal. A final list of subjects to use in first-level analyses is created.</td><td>completed</td><td>None</td><td>For time series analyses used subjects listed in preanalysis_good_subject_list.txt (in associated analysis directory).   A mask for group (randomise) analyses is created in this code base (see other files section for mask location.)</td></tr>
<tr><td>create events modified</td><td>Modified the events files in order to make all responses on a scale of 0–1, and code the responses such that a higher response corresponded to greater self-regulation.</td><td>completed</td><td>None</td><td>None</td></tr>
<tr><th colspan="5" style="background-color:#f2f2f2; font-size:1.1em;">Time Series</th></tr>
<tr><th>ID</th><th>Description</th><th>Status</th><th>Results Files</th><th>Notes</th></tr>
<tr><td>within subject brain behavior analysis</td><td>Estimate the average of the within-subject correlation between brain  activation and behavioral response across all questions.</td><td>completed</td><td>None</td><td>None</td></tr>
<tr><td>within subject question estimates</td><td>This analysis runs time series models for all subjects to estimate question-specific estimates as well as questionnaire-specific estimates. For the questionnaire averages, all question estimates were included, regardless of whether the subject responded or had an acceptable key press.   Contrast estimates filenames include "err" if they did not respond or used an unacceptable key for that question.</td><td>complete</td><td>None</td><td>There are lists in text files indicating the good subjects for each contrast (based on the outlier assessment)  located in the outlier_assessment subdirectory within the output directory.</td></tr>
<tr><th colspan="5" style="background-color:#f2f2f2; font-size:1.1em;">Time Series And Group</th></tr>
<tr><th>ID</th><th>Description</th><th>Status</th><th>Results Files</th><th>Notes</th></tr>
<tr><td>within subject brain behavior by questionnaire</td><td>Estimate the within-subject slope between brain activation and behavioral response separately for each questionnaire.</td><td>completed</td><td>None</td><td>None</td></tr>
<tr><th colspan="5" style="background-color:#f2f2f2; font-size:1.1em;">Group</th></tr>
<tr><th>ID</th><th>Description</th><th>Status</th><th>Results Files</th><th>Notes</th></tr>
<tr><td>questionnaire brain behavior slope ftest</td><td>Run an F-test on the within-subject slopes between brain activation and behavioral response for each questionnaire,  to see whether any within-questionnaire slopes are statistically different from each other.</td><td>ongoing</td><td>None</td><td>None</td></tr>
<tr><td>questionnaire average omnibus f</td><td>We use randomise to run an omnibus f-test to test whether any pair of questionaire average activations differ from each other.  The permutation scheme adjusts for the correlations between measures within-subject in the group model (model mean adjusts by subject as well).  Individual paired t-tests for all 10 pairs of  questionnaires are also run in an effort to help understand the f-test results.</td><td>Results files needs review by Patrick</td><td><a href="../analyses/questionnaire_average_omnibus_f/review_results.ipynb">review_results.ipynb</a></td><td>The paired comparisons can be used to conclude where differences  DO occur, but a lack of a significant paired t-test cannot be used to conclude two questionnaires are the same.  Keep in mind that questionnaires with fewer questions have less power in these comparisons.</td></tr>
</table>
---

## Detailed Reports


## Preprocessing

### assess_subject_bold_dropout
**Name:** Check nifti masker mask overlap to assess subject dropout<br>
**Description:** The overlap of the individual masker masks with a generous aggregate mask (voxels with data for at least 30% of the subjects) is used to identify subjects with poor coverage that may be considered for  removal. A final list of subjects to use in first-level analyses is created.<br>
**Code Directory:** analyses/assess_subject_bold_dropout<br>
**Dependencies:** None<br>
**Script Entry:** visualize_masker_on_mean_bold.py, submit_visualize_masker_on_mean_bold.py<br>
**Notebook Entry:** generate_good_subject_list.ipynb<br>
**Other Files:** analyses/assess_subject_bold_dropout/preanalysis_good_subject_list.txt, group mask for randomise: .../survey_medley_results/assess_subject_bold_dropout/group_mask_intersection_30pct.nii.gz<br>
**Output Directory:** /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/assess_subject_bold_dropout/nifti_masker_masks<br>
**Results Files:** None<br>
**Hypothesis:** Some subjects with poor data may have been missed during MRIQC evaluation.<br>
**Conclusion:** Two subjects were identified for removal due to poor coverage caused by bad data quality (see notebook for details).<br>
**Notes:** For time series analyses used subjects listed in preanalysis_good_subject_list.txt (in associated analysis directory).   A mask for group (randomise) analyses is created in this code base (see other files section for mask location.)<br>
**Status:** completed<br>
**Last Updated:** 2025-11-23<br>
**Authors:** Jeanette Mumford<br>

---

### create_events_modified
**Name:** Modify events files to create consistent scale and adjust question codings<br>
**Description:** Modified the events files in order to make all responses on a scale of 0–1, and code the responses such that a higher response corresponded to greater self-regulation.<br>
**Code Directory:** analyses/create_events_modified<br>
**Dependencies:** None<br>
**Script Entry:** fix_codings.py<br>
**Notebook Entry:** None<br>
**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** None<br>
**Hypothesis:** None<br>
**Conclusion:** None<br>
**Notes:** None<br>
**Status:** completed<br>
**Last Updated:** 2025-11-18<br>
**Authors:** Jaemon Jumpawong<br>

---


## Time Series

### within_subject_brain_behavior_analysis
**Name:** Assess within-subject brain-behavior correlations<br>
**Description:** Estimate the average of the within-subject correlation between brain  activation and behavioral response across all questions.<br>
**Code Directory:** analyses/within_subject_brain_behavior_analysis<br>
**Dependencies:** None<br>
**Script Entry:** within_subject_brain_behavior_outlier_analysis.py, run_outlier_analysis.batch, submit_randomise.sh<br>
**Notebook Entry:** analyze_brain_behavior_within_subject.ipynb<br>
**Other Files:** None<br>
**Output Directory:** oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_analysis<br>
**Results Files:** None<br>
**Hypothesis:** Within-subject brain activation linearly relates to their behavioral response.<br>
**Conclusion:** There are regions with a significant negative correlation between brain activation and behavioral response (across all questions), suggesting that these regions are more activated when giving a response coded as less self-regulated.<br>
**Notes:** None<br>
**Status:** completed<br>
**Last Updated:** 2025-11-16<br>
**Authors:** Jaemon Jumpawong, Jeanette Mumford<br>

---

### within_subject_question_estimates
**Name:** Estimate question and questionnaire activation estimates and QA<br>
**Description:** This analysis runs time series models for all subjects to estimate question-specific estimates as well as questionnaire-specific estimates. For the questionnaire averages, all question estimates were included, regardless of whether the subject responded or had an acceptable key press.   Contrast estimates filenames include "err" if they did not respond or used an unacceptable key for that question.<br>
**Code Directory:** analyses/within_subject_question_estimates<br>
**Dependencies:** None<br>
**Script Entry:** timeseries_model.py<br>
**Notebook Entry:** make_subid_text_file.ipynb, qa_outputs.ipynb<br>
**Other Files:** None<br>
**Output Directory:** /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_question_estimates/within_subject_results<br>
**Results Files:** None<br>
**Hypothesis:** No formal hypothesis at this stage; just calculating estimates for group analyses to study questionnaire differences.<br>
**Conclusion:** The outlier assessment worked well and we have at least 93 good data sets for each question/questionnaire.<br>
**Notes:** There are lists in text files indicating the good subjects for each contrast (based on the outlier assessment)  located in the outlier_assessment subdirectory within the output directory.<br>
**Status:** complete<br>
**Last Updated:** 2025-11-20<br>
**Authors:** Jeanette Mumford, Jaemon Jumpawong<br>

---


## Time Series And Group

### within_subject_brain_behavior_by_questionnaire
**Name:** Assess within-subject brain-behavior slopes in each questionnaire<br>
**Description:** Estimate the within-subject slope between brain activation and behavioral response separately for each questionnaire.<br>
**Code Directory:** analyses/within_subject_brain_behavior_by_questionnaire<br>
**Dependencies:** None<br>
**Script Entry:** questionnaire_brain_behavior_contrasts_outlier_analysis.py, run_outlier_analysis.batch<br>
**Notebook Entry:** brain_behavior_within_subject_by_questionnaire.ipynb<br>
**Other Files:** None<br>
**Output Directory:** oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_brain_behavior_by_questionnaire<br>
**Results Files:** None<br>
**Hypothesis:** Some brain/behavior slopes are statistically different between questionnaires.<br>
**Conclusion:** Still under discussion, but the F-test yielded some significant regions,  some of which overlapped with the regions found in the within-subject brain-behavior analysis with all questions.<br>
**Notes:** None<br>
**Status:** completed<br>
**Last Updated:** 2025-11-16<br>
**Authors:** Jaemon Jumpawong, Jeanette Mumford<br>

---


## Group

### questionnaire_brain_behavior_slope_ftest
**Name:** Assess how within-subject brain-behavior slopes differ between questionnaires<br>
**Description:** Run an F-test on the within-subject slopes between brain activation and behavioral response for each questionnaire,  to see whether any within-questionnaire slopes are statistically different from each other.<br>
**Code Directory:** analyses/questionnaire_brain_behavior_slope_ftest<br>
**Dependencies:** analyses/within_subject_brain_behavior_by_questionnaire<br>
**Script Entry:** None<br>
**Notebook Entry:** run_ftest.ipynb<br>
**Other Files:** None<br>
**Output Directory:** oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/questionnaire_brain_behavior_slope_ftest<br>
**Results Files:** None<br>
**Hypothesis:** Some within-questionnaire slopes statistically differ from each other in some parts of the brain.<br>
**Conclusion:** Still under discussion, but the F-test yielded some significant regions,  some of which overlapped with the regions found in the within-subject brain-behavior analysis with all questions.<br>
**Notes:** None<br>
**Status:** ongoing<br>
**Last Updated:** 2025-11-16<br>
**Authors:** Jaemon Jumpawong, Jeanette Mumford<br>

---

### questionnaire_average_omnibus_f
**Name:** Run and process omnibus f comparing all pairs of questionnaire average activation estimates<br>
**Description:** We use randomise to run an omnibus f-test to test whether any pair of questionaire average activations differ from each other.  The permutation scheme adjusts for the correlations between measures within-subject in the group model (model mean adjusts by subject as well).  Individual paired t-tests for all 10 pairs of  questionnaires are also run in an effort to help understand the f-test results.<br>
**Code Directory:** analyses/questionnaire_average_omnibus_f<br>
**Dependencies:** Output files from within_subject_question_estimates analysis<br>
**Script Entry:** randomise scripts are created in group_ftest and each subdirectory of all_paired_t_tests in survey_medley_results/within_subject_question_estimates/<br>
**Notebook Entry:** setup_randomise.ipynb, review_results.ipynb<br>
**Other Files:** runs f-test randomise script: submit_randomise.batch, runs individual paired randomise tests: submis_randomise_all_pairs.batch<br>
**Output Directory:** /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/within_subject_question_estimates/within_subject_question_estimates/<br>
**Results Files:** None<br>
**Hypothesis:** Some questionnaire-based average activation estimates will differ from others.<br>
**Conclusion:** There is a strong result for the omnibus f-test, but Patrick needs to review the individual paired comparisons.<br>
**Notes:** The paired comparisons can be used to conclude where differences  DO occur, but a lack of a significant paired t-test cannot be used to conclude two questionnaires are the same.  Keep in mind that questionnaires with fewer questions have less power in these comparisons.<br>
**Status:** Results files needs review by Patrick<br>
**Last Updated:** 2025-11-25<br>
**Authors:** Jeanette Mumford, Jaemon Jumpawong<br>

---
