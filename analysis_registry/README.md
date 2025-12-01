# Master Analysis Registry

## Summary

<table>
<tr><th>ID</th><th>Description</th><th>Status</th><th>Notes</th></tr>
<tr><td colspan="4"><strong>Preprocessing</strong></td></tr>
<tr><td>assess subject bold dropout</td><td>The overlap of the individual masker masks with a generous aggregate mask (voxels with data for at least half of the subjects) is used to identify subjects with poor coverage that may be considered for  removal. A final list of subjects to use in first-level analyses is created.</td><td>completed</td><td>Start analyses with subjects in preanalysis_good_subject_list.txt (in associated analysis directory)</td></tr>
<tr><td>create events modified</td><td>Modified the events files in order to make all responses on a scale of 0–1, and code the responses such that a higher response corresponded to greater self-regulation.</td><td>completed</td><td>None</td></tr>
<tr><td colspan="4"><strong>Time Series</strong></td></tr>
<tr><td>within subject brain behavior analysis</td><td>Estimate the average of the within-subject correlation between brain  activation and behavioral response across all questions.</td><td>completed</td><td>None</td></tr>
<tr><td>within subject question estimates</td><td>This analysis runs time series models for all subjects to estimate question-specific estimates as well as questionnaire-specific estimates. For the questionnaire averages, all question estimates were included, regardless of whether the subject responded or had an acceptable key press.   Contrast estimates filenames include "err" if they did not respond or used an unacceptable key for that question.</td><td>complete</td><td>There are lists in text files indicating the good subjects for each contrast (based on the outlier assessment)  located in the outlier_assessment subdirectory within the output directory.</td></tr>
<tr><td colspan="4"><strong>Time Series And Group</strong></td></tr>
<tr><td>within subject brain behavior by questionnaire</td><td>Estimate the within-subject slope between brain activation and behavioral response separately for each questionnaire.</td><td>completed</td><td>None</td></tr>
<tr><td colspan="4"><strong>Group</strong></td></tr>
<tr><td>questionnaire brain behavior slope ftest</td><td>Run an F-test on the within-subject slopes between brain activation and behavioral response for each questionnaire,  to see whether any within-questionnaire slopes are statistically different from each other.</td><td>ongoing</td><td>None</td></tr>
</table>

---

## Detailed Reports


## Preprocessing

### assess_subject_bold_dropout
**Name:** Check nifti masker mask overlap to assess subject dropout<br>
**Description:** The overlap of the individual masker masks with a generous aggregate mask (voxels with data for at least half of the subjects) is used to identify subjects with poor coverage that may be considered for  removal. A final list of subjects to use in first-level analyses is created.<br>
**Code Directory:** analyses/assess_subject_bold_dropout<br>
**Dependencies:** None<br>
**Script Entry:** visualize_masker_on_mean_bold.py, submit_visualize_masker_on_mean_bold.py<br>
**Notebook Entry:** generate_good_subject_list.ipynb<br>
**Other Files:** analyses/assess_subject_bold_dropout/preanalysis_good_subject_list.txt<br>
**Output Directory:** /oak/stanford/groups/russpold/data/uh2/aim1/derivatives/survey_medley_results/assess_subject_bold_dropout/nifti_masker_masks<br>
**Hypothesis:** Some subjects with poor data may have been missed during MRIQC evaluation.<br>
**Conclusion:** Two subjects were identified for removal due to poor coverage caused by bad data quality (see notebook for details).<br>
**Notes:** Start analyses with subjects in preanalysis_good_subject_list.txt (in associated analysis directory)<br>
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
**Hypothesis:** Some within-questionnaire slopes statistically differ from each other in some parts of the brain.<br>
**Conclusion:** Still under discussion, but the F-test yielded some significant regions,  some of which overlapped with the regions found in the within-subject brain-behavior analysis with all questions.<br>
**Notes:** None<br>
**Status:** ongoing<br>
**Last Updated:** 2025-11-16<br>
**Authors:** Jaemon Jumpawong, Jeanette Mumford<br>

---
