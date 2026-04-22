---
name: sklearn-modeling
description: Use this skill when training, evaluating, and tuning scikit-learn classification models. Follow the baseline-first approach with cross-validation, then tune the best model and generate a submission file.
---

# Scikit-learn Modeling Workflow

## Overview

Systematic ML model training and evaluation workflow for tabular classification.
Follows the baseline-first principle: establish baselines → compare → tune best → generate submission.

## When to Use

- After feature engineering is complete and processed data is ready
- When the orchestrator asks you to "train models", "build a classifier", or "generate predictions"
- When `X_train.csv`, `y_train.csv`, and `X_test.csv` exist in `/home/gem/data/`

## Best Practices

- **Baselines first** — Always establish baselines before tuning
- **Cross-validation** — Never rely on a single train/test split; use 5-fold CV
- **Print comparison table** — The orchestrator needs a clear summary to report to the user
- **Save results** — Write model comparison to `/home/gem/reports/model_results.md`
- **Correct submission format** — Match Kaggle's expected format exactly

## Process

### Step 1: Load Processed Data

```python
import pandas as pd
import numpy as np

X_train_df = pd.read_csv('/home/gem/data/X_train.csv')
y_train = pd.read_csv('/home/gem/data/y_train.csv').values.ravel()
X_test_df = pd.read_csv('/home/gem/data/X_test.csv')

# Preserve IDs for submission
test_ids = X_test_df['PassengerId']

# Drop PassengerId from features (it's not a predictive feature)
X_train = X_train_df.drop(columns=['PassengerId']).values
X_test = X_test_df.drop(columns=['PassengerId']).values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Target distribution: {np.bincount(y_train)}")
```

### Step 2: Train Baseline Models

Train multiple baseline models with default hyperparameters.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n--- Baseline Model Comparison (5-Fold CV) ---")
print(f"{'Model':<25} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 47)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>10.4f}")
```

### Step 3: Select Best Model

Identify the top-performing model by CV mean.

```python
best_model_name = max(results, key=lambda k: results[k]['mean'])
best_score = results[best_model_name]['mean']
print(f"\nBest baseline: {best_model_name} (CV accuracy: {best_score:.4f})")
```

### Step 4: Hyperparameter Tuning

Tune the best model's hyperparameters using GridSearchCV or RandomizedSearchCV.

```python
from sklearn.model_selection import RandomizedSearchCV

# Example: Tune GradientBoosting (adjust param_distributions for your best model)
if best_model_name == 'GradientBoosting':
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
    }
    base = GradientBoostingClassifier(random_state=42)

elif best_model_name == 'RandomForest':
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    base = RandomForestClassifier(random_state=42)

else:
    param_distributions = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
    }
    base = LogisticRegression(max_iter=1000, random_state=42)

search = RandomizedSearchCV(
    base,
    param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)

print(f"\nTuned {best_model_name}:")
print(f"  Best CV accuracy: {search.best_score_:.4f}")
print(f"  Best params: {search.best_params_}")
print(f"  Improvement: {search.best_score_ - best_score:+.4f}")
```

### Step 5: Generate Submission

Use the tuned model to predict on test set and create submission file.

```python
import os

# Use the tuned model (already fitted on full train set by RandomizedSearchCV)
final_model = search.best_estimator_
predictions = final_model.predict(X_test)

# Format submission for Kaggle
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': predictions.astype(bool),  # Convert 0/1 back to True/False
})

os.makedirs('/home/gem/output', exist_ok=True)
submission.to_csv('/home/gem/output/submission.csv', index=False)

print(f"\n--- Submission ---")
print(f"Shape: {submission.shape}")
print(f"First 5 rows:\n{submission.head()}")
print(f"Distribution: {submission['Transported'].value_counts().to_dict()}")
print(f"Saved to: /home/gem/output/submission.csv")
```

### Step 6: Save Model Results Report

Write a structured report for the orchestrator.

```python
report = f"""# Model Training Results

## Baseline Comparison (5-Fold CV, Accuracy)

| Model | CV Mean | CV Std |
|---|---|---|
"""
for name, res in sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True):
    report += f"| {name} | {res['mean']:.4f} | {res['std']:.4f} |\n"

report += f"""
## Tuned Model

- **Model:** {best_model_name}
- **Best CV Accuracy:** {search.best_score_:.4f}
- **Improvement over baseline:** {search.best_score_ - best_score:+.4f}
- **Best Parameters:** {search.best_params_}

## Submission

- File: `/home/gem/output/submission.csv`
- Shape: {submission.shape}
- Prediction distribution: {submission['Transported'].value_counts().to_dict()}
"""

os.makedirs('/home/gem/reports', exist_ok=True)
with open('/home/gem/reports/model_results.md', 'w') as f:
    f.write(report)
print("Report saved to /home/gem/reports/model_results.md")
```

## Anti-Patterns

- **Never** tune before establishing baselines — you need a reference point
- **Never** use test labels for evaluation — we don't have them; use cross-validation
- **Never** skip cross-validation — a single train/test split is unreliable
- **Never** forget to print the comparison table — the orchestrator needs it for HITL reporting
- **Never** submit predictions as 0/1 integers if the format expects True/False strings

## Common Pitfalls

- **Forgetting `random_state`** — makes results non-reproducible
- **Not setting `max_iter` for LogisticRegression** — convergence warning
- **Using `n_jobs=-1` on small datasets** — parallel overhead exceeds benefit
- **Not preserving PassengerId** — submission will be rejected
- **Training on all data before CV** — data leakage through feature selection or scaling
