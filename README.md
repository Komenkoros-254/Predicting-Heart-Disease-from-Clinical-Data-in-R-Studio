# Heart Disease Prediction in R

Predicting the presence of heart disease from 13 clinical measurements, and finding out which measurements matter most. Four models are compared with repeated cross-validation, and an interpretable logistic regression is evaluated on held-out patients. Built in R with tidymodels.

> **Disclaimer:** this is a learning project. It is not validated for clinical use and must not guide medical decisions.

## Key results

- **Four models perform almost identically.** Cross-validated ROC-AUC ranged from 0.895 to 0.918, and the top three models are statistically indistinguishable. A simple logistic regression gives up little accuracy, so it was chosen as the final model for interpretability.
- **On unseen patients** the final model reached ROC-AUC 0.903, accuracy 87%, sensitivity 79% and specificity 93% (default 0.5 threshold, 54 test patients).
- **Strongest risk signals:** number of blocked major vessels, asymptomatic chest pain, male sex, a reversible thalassemia defect, ST depression and a flat ST slope. Cholesterol, resting blood pressure and fasting blood sugar add little.
- **The random forest agrees:** its permutation importance ranks reversible thal defect, blocked vessels, asymptomatic chest pain and ST depression as the top four predictors.
- **The model misses a recognisable group:** patients whose disease shows no ischaemic signs on testing (no blocked vessels, no ST depression).

### Model comparison (repeated 5-fold CV, 25 resamples)

| Model | ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Random forest | 0.918 | 0.842 | 0.774 | 0.895 |
| Lasso logistic | 0.915 | 0.857 | 0.787 | 0.913 |
| XGBoost | 0.911 | 0.844 | 0.809 | 0.872 |
| Logistic regression | 0.895 | 0.838 | 0.789 | 0.877 |

![Model comparison](figures/model_comparison.png)

### Odds ratios for disease (logistic regression, significant predictors)

| Predictor | Odds ratio | 95% CI | p |
|---|---|---|---|
| Blocked vessels (`ca`), per 1 SD | 3.94 | 2.17 to 7.94 | <0.001 |
| Asymptomatic vs typical chest pain | 2.70 | 1.19 to 6.58 | 0.021 |
| Male vs female | 2.45 | 1.36 to 4.76 | 0.005 |
| Reversible vs normal thal result | 2.01 | 1.24 to 3.32 | 0.005 |
| ST depression (`oldpeak`), per 1 SD | 1.93 | 1.06 to 3.73 | 0.039 |
| Flat vs upsloping ST slope | 1.83 | 1.06 to 3.26 | 0.033 |

Numeric predictors were standardised, so their odds ratios are per 1 standard deviation.



## Data

270 patients, 13 predictors and a binary target (1 = heart disease, 0 = no disease). 120 patients (44.4%) have disease. The data appears to be the 270-patient Statlog version of the classic Cleveland heart disease dataset. See [`data/README.md`](data/README.md) for the column dictionary.

Source: https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog

## Method

1. **Cleaning:** removed duplicates before splitting, recoded invalid codes (`ca = 4`, `thal = 0`) as missing, and converted categorical variables to labelled factors.
2. **Target-coding check:** some public copies of this dataset have the target flipped. Disease rates were checked against clinical logic before interpretation (rates rise with the number of blocked vessels and with exercise angina), confirming 1 = disease.
3. **Exploratory analysis:** disease rate by category, numeric predictors by outcome, correlations, and rates by age band and sex.
4. **Split and resampling:** 80/20 stratified split (216 training, 54 test patients); stratified 5-fold cross-validation repeated 5 times.
5. **Preprocessing:** median and mode imputation, dummy encoding, zero-variance removal and standardisation, all fitted inside each resample to avoid leakage.
6. **Models:** logistic regression, lasso, random forest and XGBoost, with hyperparameters tuned over a 15-point grid.
7. **Evaluation:** ROC-AUC, accuracy, sensitivity and specificity; threshold analysis favouring recall; calibration plot; odds ratios; permutation importance; error analysis of false negatives.

## Repository structure

```
heart-disease-prediction/
├── README.md
├── heart-disease-prediction.Rproj
├── renv.lock
├── data/
│   ├── raw/heart.csv
│   └── README.md
├── R/
│   └── heart_disease_analysis.R
├── figures/
└── reports/
    └── heart_disease_report.pdf
```

## How to run

1. Clone the repository and open `heart-disease-prediction.Rproj` in RStudio.
2. Restore the package versions used in the analysis:
   ```r
   renv::restore()
   ```
3. Run `R/heart_disease_analysis.R` section by section. The data is read from `data/raw/heart.csv`.

Main packages: `tidymodels`, `tidyverse`, `janitor`, `glmnet`, `ranger`, `xgboost`, `probably`, `corrplot`. The full XGBoost and random forest tuning takes several minutes; reduce `repeats` or `grid` in the script to speed it up.

## Limitations

- **Small sample:** 270 patients and a 54-patient test set, so test-set metrics are noisy and confidence intervals are wide. Cross-validated results are the more reliable headline numbers.
- **Single source, no external validation:** results may not transfer to other hospitals or populations.
- **Referral bias:** all patients were referred for testing, which explains findings such as the high disease rate among patients with asymptomatic chest pain.
- **Sex imbalance:** 183 men and 87 women, so estimates for women are less certain.
- **Optimism from tuning:** models were tuned and compared on the same resamples, so cross-validated scores are slightly optimistic.
- **Association, not causation:** odds ratios describe patterns in this data, not causal effects.


## Author

Komen Kipkoros Meshack · https://www.datascienceportfol.io/komenkoros1999 · komenkoros1999@gmail


