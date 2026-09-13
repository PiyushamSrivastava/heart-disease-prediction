# Heart Disease Risk Prediction

Predicting cardiovascular disease risk from routine clinical and lifestyle data using machine learning, with a focus on minimizing missed diagnoses (false negatives).

## Dataset

[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — 70,000 patient records with 11 features:

- **Demographic**: age, gender, height, weight
- **Clinical**: systolic blood pressure (sbp), diastolic blood pressure (dbp), cholesterol, glucose
- **Lifestyle**: smoking, alcohol intake, physical activity
- **Target**: `cardio` (presence of cardiovascular disease)

## Data Cleaning

- Removed non-predictive `id` column
- Converted `age` from days to years
- Renamed `ap_hi`/`ap_lo` to `sbp`/`dbp` for clarity
- Identified and handled implausible outliers in blood pressure, height, and weight

## Modeling Approach

Three models were trained and compared on a held-out 20% test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.720 | 0.755 | 0.653 | 0.700 | 0.775 |
| Random Forest (tuned) | 0.732 | 0.765 | 0.670 | 0.714 | 0.785 |
| XGBoost (default) | 0.731 | 0.765 | 0.668 | 0.713 | 0.784 |
| **XGBoost (RandomizedSearchCV tuned)** | **0.740** | **0.761** | **0.701** | **0.730** | **0.803** |

All three algorithms converged to a similar performance ceiling (~73-74% accuracy), suggesting the raw feature set — not model choice — was the limiting factor.

## Hyperparameter Tuning

Used `RandomizedSearchCV` (20 iterations, 5-fold cross-validation) optimizing directly for **recall**, since missing a true positive case (an undiagnosed heart disease patient) carries a higher real-world cost than a false alarm.

Best parameters found:
```
subsample: 0.8, n_estimators: 300, max_depth: 5,
learning_rate: 0.05, gamma: 1, colsample_bytree: 0.8
```

## Threshold Optimization

Rather than using the default 0.5 classification threshold, a precision-recall curve was analyzed to select a threshold better aligned with the cost asymmetry of this problem.

| Threshold | Accuracy | Precision | Recall | False Negatives |
|---|---|---|---|---|
| 0.5 (default) | 0.740 | 0.761 | 0.701 | 2,095 |
| **0.4 (selected)** | **0.727** | **0.702** | **0.790** | **1,474** |
| 0.35 | 0.709 | 0.665 | 0.842 | 1,108 |

**Threshold 0.4 was selected as the final model** — it reduces false negatives by ~30% versus the default threshold while keeping the precision trade-off more moderate than a more aggressive threshold (0.35).

## Feature Engineering (Tested, Not Adopted)

Additional clinical features were engineered and tested, including BMI, a categorical blood pressure bucket (`bp_category`), pulse pressure, age groups, and a lifestyle risk score.

- `bp_category` became the single most important feature in the model (43.6% importance) when included.
- Despite this, overall performance was statistically unchanged (accuracy 0.7404 vs 0.740, ROC-AUC 0.803 vs 0.803).
- **Conclusion**: tree-based models like XGBoost can already learn non-linear threshold splits on raw blood pressure values, so manual bucketing added redundant, not new, information. The raw feature set was retained for the final model to keep the pipeline simpler without sacrificing performance.

## Feature Importance (Final Model)

| Feature | Importance |
|---|---|
| sbp | 39.2% |
| cholesterol | 17.9% |
| dbp | 14.0% |
| age | 9.6% |
| smoke | 3.7% |
| gluc | 3.5% |
| active | 3.1% |
| weight | 2.6% |
| alco | 2.6% |
| gender | 2.0% |
| height | 1.7% |

Blood pressure, cholesterol, and age dominate the model's predictions — consistent with established clinical risk factors for cardiovascular disease.

## Final Model

- **Algorithm**: XGBoost, tuned via RandomizedSearchCV
- **Features**: 11 raw clinical/demographic/lifestyle features
- **Decision threshold**: 0.4 (not default 0.5)
- **Performance**: 72.7% accuracy, 70.2% precision, 79.0% recall, 74.3% F1, 80.3% ROC-AUC

## Key Takeaways

1. Model choice mattered less than expected — Logistic Regression, Random Forest, and XGBoost all plateaued around the same performance ceiling.
2. Hyperparameter tuning with a recall-focused objective produced a meaningful, measurable improvement.
3. Threshold selection is a deliberate design decision, not a default to leave untouched — especially in domains where error types carry different real-world costs.
4. Not all feature engineering helps: bucketed blood pressure categories didn't improve a tree-based model that could already learn those splits on its own — a useful negative result.

## Tech Stack

`pandas`, `scikit-learn`, `xgboost`, `matplotlib`
