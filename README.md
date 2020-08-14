# Federated Online Learning for Time Series Data

Machine learning for time series data benefits from continuous training of the model (online learning). The same is true for federated learning, but this comes with an additional communication overhead.

This project wants to **study the impact online learning for time series data has in federated settings**. In particular **how to maximise the accuracy** of the system, while **minimising the number of gradient updates** with the central server (due to **communication overhead**, but also due to **potential privacy losses due to frequent gradient updates**)

## Paper + Dataset Information

- **Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection** ([Source](https://dl.acm.org/doi/abs/10.1145/3242969.3242985))

WESAD (Wearable Stress and Affect Detection) is a [publicly available](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) dataset for wearable stress and affect detection. This [multimodal](https://stats.stackexchange.com/a/168591) dataset features **physiological and motion** data recorded from both a wrist and a chest-worn device:

* **Source**
    * 15 subjects exposed to affective stimuli (stress and amusement)
        * 12(M) + 3(F)
        * ages: 27.5 +- 2.5 years
        * ~ 36 min per subject
        * Segmentation using a sliding window of 0.25sec -> ~133000 windows generated
            * 53% from baseline
            * 30% from stress
            * 17% from amusement
    * Wrist and Chest-worn devices
    * Self-reports of the subjects (questionnaires)
* **Sensor modalities**
    * 81 total features among modalities from both devices:
       * Blood volume pulse (BVP)
       * Electrocardiogram (ECG)
       * Eletrodermal activity (EDA)
       * Electromyogram (EMG)
       * Respiration (RESP)
       * Body temperature (TEMP)
       * Three axes accelerometer (ACC)
* **Three different affective states**
    * Baseline (Neutral)
    * Stress
    * Amusement
* **Benchmark using different Classification algorithms**
    * Three-class classification problem: *baseline vs stress vs amusement*
    * Binary classification: stress vs non-stress (baseline+amusement)
    * Evaluation using leave-one-subject-out (LOSO) CV procedure
    * Evaluation metrics: Accuracy & F1-Score using averaged precision and recall over the three classes
    * Algorithms:
      - Decision Tree (DT)
      - Random Forest (RF)
      - AdaBoost (AB)
      - Linear Discriminant Analysis (LDA)
      - k-Nearest Neighbours (kNN)
    * Hyperparameters:
      - n_estimators = 100
      - min_samples_split = 20
      - criterion = 'entropy' (information gain)

## Reproducing Paper results

See [BASELINE_EXPERIMENTS.md](BASELINE_EXPERIMENTS.md) for information on reproducing dataset feature extraction + ML models benchmark.

## Baseline (non-federated) models

### LSTM

* Check notebook: [LSTM_Model_Chest_device.ipynb](LSTM_Model_Chest_device.ipynb) (**WIP**)

### CNN

* WIP
