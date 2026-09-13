# Heart Disease Risk Prediction

A machine learning project to predict cardiovascular disease risk from routine clinical and lifestyle data. Built with a focus on catching as many real cases as possible, since missing a diagnosis is worse than a false alarm.

## Dataset

[Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle — 70,000 records with age, gender, height, weight, blood pressure, cholesterol, glucose, and a few lifestyle habits (smoking, alcohol, activity level).

## What I did

Started by cleaning the data — dropped the useless `id` column, converted age from days to years, renamed the blood pressure columns to `sbp`/`dbp` so they actually make sense, and handled some obvious outliers (things like blood pressure readings of 0 or height listed as 50cm).

Then I trained three models to compare: Logistic Regression, Random Forest, and XGBoost. All three landed around 72-74% accuracy, which told me pretty quickly that the bottleneck wasn't the algorithm — it was the data itself. Eleven features (age, blood pressure, cholesterol, a few lifestyle flags) can only tell you so much about someone's heart disease risk without genetics, diet, or family history.

I tuned XGBoost using RandomizedSearchCV, optimizing specifically for recall instead of accuracy, since in a health context, missing an actual case matters more than a false positive. That pushed recall from 65% to 70% and ROC-AUC to 0.80.

I also tried engineering a few extra features — BMI, a categorical blood pressure bucket, pulse pressure. The blood pressure bucket ended up being the single most important feature by importance score, but it didn't actually move the needle on any of the metrics. Made sense once I thought about it: XGBoost can already split on raw blood pressure thresholds on its own, so manually bucketing it didn't teach it anything new. Kept the simpler raw-feature model instead of adding complexity for no gain.

Last step was tuning the decision threshold. Instead of the default 0.5 cutoff, I looked at the precision-recall trade-off curve and moved it down to 0.4 — this cut missed cases by about 30% while only giving up a moderate amount of precision.

## Results

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 72.0% | 75.5% | 65.3% | 70.0% | 0.775 |
| Random Forest | 73.2% | 76.5% | 67.0% | 71.4% | 0.785 |
| XGBoost (tuned) | 74.0% | 76.1% | 70.1% | 73.0% | 0.803 |
| **XGBoost (tuned + threshold 0.4)** | **72.7%** | **70.2%** | **79.0%** | **74.3%** | **0.803** |

The final model trades a bit of precision for a meaningful recall gain — going from missing ~2,095 real cases at the default threshold down to ~1,474.

## What actually mattered

Feature importance confirmed the model leans on blood pressure, cholesterol, and age above everything else — which lines up with what's already known clinically about heart disease risk factors. Things like height and gender barely moved the needle, and dropping them didn't hurt performance at all.

## Tech stack

Python, pandas, scikit-learn, XGBoost, matplotlib

## Notes

The raw dataset isn't included here — grab it from the Kaggle link above. `requirements.txt` has everything needed to reproduce the environment.
