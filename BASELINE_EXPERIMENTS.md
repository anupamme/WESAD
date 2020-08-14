
# Baseline Experiments

This document presents a brief summary of reproducing results from the paper **Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection**

## Feature Extraction

* Check Notebook: [WESAD_Data_Exploration.ipynb](WESAD_Data_Exploration.ipynb)
* 23,206.404 total records in preprocessed sensor signals prior segmentation

### Chest-worn device

- Segmentation of the (preprocessed) sensor signals was done using a 60-second sliding window, with a window shift of 0.25 seconds and default sampling rate (700 Hz). A total of 79 (out of 81) features were computed. A summary of total windows obtained in the process is shown below

|Subject|Windows|
|------:|------:|
|  S2   |   7764|
|  S3   |   7900|
|  S4   |   7941|
|  S5   |   8148|
|  S6   |   8088|
|  S7   |   8073|
|  S8   |   8116|
|  S9   |   8068|
| S10   |   8388|
| S11   |   8192|
| S13   |   8185|
| S14   |   8189|
| S15   |   8212|
| S16   |   8165|
| S17   |   8384|
|**TOTAL**| 121813|

### Wrist-worn device

* WIP

## Reproducing benchmark experiments

### Using Chest-worn device data

- We trained a Random Forest (RF), AdaBoost (AB) and Latent Discriminant Analysis (LDA) models suing different sets of sensor modalities.
- Each setup was run five times so mean and std of scores are reported.
- Final scores within each experiment were averaged using a LOSO (Leave One Subject Out) cross-validation.

1. Evaluation using each of the six sensor modalities separately: 

|      |    rf_f1 |   rf_acc |   rf_b_f1 |   rf_b_acc |    ab_f1 |   ab_acc |   ab_b_f1 |   ab_b_acc |   lda_f1 |   lda_acc |   lda_b_f1 |   lda_b_acc |
|:-----|---------:|---------:|----------:|-----------:|---------:|---------:|----------:|-----------:|---------:|----------:|-----------:|------------:|
| ACC  | 0.395361 | 0.559278 |  0.569841 |   0.723911 | 0.448804 | 0.591598 |  0.588321 |   0.733027 | 0.400382 |  0.597001 |   0.542222 |    0.70007  |
| ECG  | 0.518569 | 0.660007 |  0.7963   |   0.839966 | 0.515433 | 0.662472 |  0.800814 |   0.8441   | 0.515813 |  0.70816  |   0.809485 |    0.865115 |
| EDA  | 0.643974 | 0.667019 |  0.749781 |   0.787745 | 0.621201 | 0.645761 |  0.741442 |   0.781814 | 0.451855 |  0.57464  |   0.692621 |    0.764252 |
| EMG  | 0.462594 | 0.607954 |  0.599067 |   0.713775 | 0.480842 | 0.620192 |  0.610554 |   0.723836 | 0.445108 |  0.599009 |   0.614172 |    0.724727 |
| RESP | 0.545764 | 0.676278 |  0.78608  |   0.824339 | 0.544442 | 0.68253  |  0.785471 |   0.821546 | 0.545973 |  0.698706 |   0.796305 |    0.841577 |
| TEMP | 0.408837 | 0.499755 |  0.53513  |   0.642769 | 0.386727 | 0.494761 |  0.500478 |   0.61475  | 0.289719 |  0.575443 |   0.430318 |    0.705284 |

\* Column names from above table have the following notation: `{model}_{binary_classifier?}_{eval_score_metric}`

- Check the notebook [ML Classifiers per Modality.ipynb](ML%20Classifiers%20per%20Modality.ipynb) for a full report on results on each LOSO cross-validation

2. Evaluation scores using **all modalities**:

|           |   mean |   std |
|:----------|-------:|------:|
| rf_f1     |  57.14 |  0.92 |
| rf_acc\*\*    |  69.07 |  0.62 |
| rf_b_f1   |  76.07 |  0.23 |
| rf_b_acc\*\*  |  **84.55** |  0.22 |
| ab_f1     |  54.38 |  0.95 |
| ab_acc    |  63.62 |  0.68 |
| ab_b_f1   |  73.35 |  0.56 |
| ab_b_acc  |  80.02 |  0.44 |
| lda_f1    |  65.31 |  -    |
| **lda_acc**\*   |  **71.63** |  -    |
| lda_b_f1  |  86.07 |  -    |
| **lda_b_acc**\* |  **89.03** |  -    |

\* Best model in terms of accuracy

\*\* Runner-up

3. Evaluation scores using **physiological modalities**:

|           |   mean |   std |
|:----------|-------:|------:|
| rf_f1     |  60.97 |  0.93 |
| rf_acc\*\*    |  **71.07** |  0.36 |
| rf_b_f1   |  78.58 |  0.61 |
| rf_b_acc\*\*  |  **85.52** |  0.47 |
| ab_f1     |  59.65 |  1.12 |
| ab_acc    |  67.47 |  0.97 |
| ab_b_f1   |  77.17 |  0.44 |
| ab_b_acc  |  82.89 |  0.47 |
| lda_f1    |  66.62 |  -    |
| **lda_acc**\*  |  **74.02** |  -    |
| lda_b_f1  |  86.43 |  -    |
| **lda_b_acc**\* |  **89.86** |  -    |

- For a more in-depth analysis of results for (2) & (3) check the notebook [ML Classifiers - Chest Device.ipynb](ML%20Classifiers%20-%20Chest%20Device.ipynb)

- Feature importance on the Three-class classification: *baseline vs stress vs amusement*

|idx | feature          |       imp |
|---:|:-----------------|----------:|
| 64 | RESP_exhal_mean  | 0.239672  |
| 36 | EDA_SCR_no       | 0.133317  |
| 15 | ECG_hr_mean      | 0.106798  |
| 56 | EMG_peak_amp_sum | 0.0847333 |
|  5 | ACC_xyz_std      | 0.0697695 |
| 74 | TEMP_min         | 0.0576274 |
| 46 | EDA_std          | 0.0369737 |
| 58 | EMG_peak_no      | 0.0243881 |
|  6 | ACC_xzy_mean     | 0.0238573 |
|  1 | ACC_x_mean       | 0.0235114 |

- Feature importance on the Binary classification: *stress vs non-stress*

|idx | feature          |       imp |
|---:|:-----------------|----------:|
| 64 | RESP_exhal_mean  | 0.395155  |
| 15 | ECG_hr_mean      | 0.157471  |
| 36 | EDA_SCR_no       | 0.139167  |
| 72 | TEMP_max         | 0.0479302 |
|  0 | ACC_x_absint     | 0.042426  |
| 58 | EMG_peak_no      | 0.0381359 |
| 44 | EDA_scr_area     | 0.0348049 |
| 65 | RESP_exhal_std   | 0.023892  |
| 56 | EMG_peak_amp_sum | 0.0164472 |
|  8 | ACC_y_mean       | 0.0162326 |

### Wrist-worn device

* WIP

