# diabetespredictor
# Diabetes Predictor

Predicts whether a patient is diabetic from 8 clinical measurements (glucose, BMI, age and others) using logistic regression. I compared it with a random forest and kept logistic regression, because it scored as well or better and is easy to explain.

> This is a learning project, not a medical tool.

## Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/) on Kaggle (search for the name): 768 patients, 8 features and a label called `Outcome` (1 = diabetic, 0 = not). The CSV is not included in this repo. Download `diabetes.csv` from Kaggle and place it next to the notebook.

## What I did

1. **Fixed two data problems.** The CSV has no header row, so pandas silently used the first patient as column names and dropped them. After fixing that, I found that `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` contain zeros that really mean "not measured" (374 of 768 patients for `Insulin`, 227 for `SkinThickness`). I converted these to missing values.
2. **Split before preprocessing.** I made an 80/20 train/test split first (stratified, so both sets have about 35% diabetic patients), then learned the median imputation and scaling from the training data only, to avoid data leakage.
3. **Set a baseline.** A model that always predicts "not diabetic" gets 64.9% accuracy, which is the number to beat.
4. **Compared two models with 5-fold cross-validation.** Using ROC AUC, logistic regression scored 0.843 and random forest 0.823. I chose logistic regression.
5. **Chose the decision threshold on training data only.** At the default 0.5 cutoff the model missed about half of the diabetic patients. Using out-of-fold predictions, I lowered the threshold to **0.4**, which catches more diabetic patients at a reasonable cost in false alarms. I favoured recall because for a screening tool, a missed patient costs more than an extra test.
6. **Evaluated once on the test set.** I looked at the held-out test set a single time, after all choices were made.
7. **Saved the final model.** The imputer, scaler and model are bundled in one scikit-learn `Pipeline`, refit on all 768 patients and saved as `diabetes_model.joblib`.

## Results (held-out test set, 154 patients)

| Metric | Value |
|---|---|
| ROC AUC | 0.813 |
| Accuracy | 0.73 |
| Diabetic patients caught (recall) | 35 of 54 (0.65) |
| Precision for "diabetic" | 0.61 |
| Baseline accuracy (always "not diabetic") | 0.649 |

Confusion matrix at threshold 0.4:

| | Predicted not diabetic | Predicted diabetic |
|---|---|---|
| **Actually not diabetic** | 78 | 22 |
| **Actually diabetic** | 19 | 35 |

Accuracy alone is misleading here. The baseline already scores 64.9% without detecting a single diabetic patient, so I focused on recall and precision.

## What the model learned

`Glucose` has by far the largest weight (1.13), followed by `BMI` (0.64) and `Pregnancies` (0.41). Since the features are standardized, a glucose level about one standard deviation (roughly 30 mg/dL) above average multiplies the odds of diabetes by about 3, all else equal. These weights describe what the model uses, not what causes diabetes.

## Limitations

- About 1 in 3 diabetic patients in the test set was still missed.
- `Insulin` and `SkinThickness` were missing for about half and a third of patients, so most of their values are filled-in medians and add little information.
- The test set is small (154 patients), so these scores are noisy. A different split could move them by a few points.
- The data covers Pima Indian women aged 21 and over, so the model may not generalize to other groups.
- The threshold of 0.4 is a judgement about costs. A real screening tool would need input from clinicians.

## How to run it

```bash
pip install -r requirements.txt
```

Then open `diabetes_predictor.ipynb` in Jupyter or Google Colab, put `diabetes.csv` next to it and run all cells.

To use the saved model on a new patient, remember that zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` mean "not measured". Convert them to `NaN` before predicting:

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("diabetes_model.joblib")

patient = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
           "SkinThickness": 0, "Insulin": 0, "BMI": 35.0,
           "DiabetesPedigreeFunction": 0.5, "Age": 45}

row = pd.DataFrame([patient])
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
row[zero_cols] = row[zero_cols].replace(0, np.nan)

prob = model.predict_proba(row)[0, 1]
print("Diabetic" if prob >= 0.4 else "Not diabetic", round(prob, 2))
```

## Tech

Python, pandas, NumPy, scikit-learn, joblib.
