---
name: eda-workflow
description: Use this skill when performing exploratory data analysis on a new tabular dataset. Follow these steps systematically to understand data shape, distributions, missing values, and feature relationships before any modeling.
---

# EDA Workflow for Tabular Data

## Overview

Systematic exploratory data analysis procedure for tabular classification datasets.
Produces a structured markdown report with key findings that guide feature engineering decisions.

## When to Use

- You receive a new dataset and need to understand its structure
- Before any feature engineering or modeling work
- When the orchestrator asks you to "analyze" or "explore" the data

## Best Practices

- **Always print results** with `print()` so output is captured by the agent
- **Process one step at a time** — do not generate all analyses in a single code block
- **Save figures to files** instead of `plt.show()` — sandbox has no display
- **Summarize findings** in plain language, not just raw numbers
- **Save the full report** to a markdown file in the sandbox for reference

## Process

### Step 1: Data Overview

Get the basic shape, types, and missing value counts.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('/home/gem/data/train.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"\n--- Column Types ---")
print(df.dtypes)
print(f"\n--- First 5 Rows ---")
print(df.head())
print(f"\n--- Basic Statistics (Numerical) ---")
print(df.describe())
print(f"\n--- Basic Statistics (Categorical) ---")
print(df.describe(include='object'))
```

### Step 2: Missing Values Analysis

Identify which columns have missing data and what percentage.

```python
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percentage', ascending=False)
print(f"\n--- Missing Values ---")
print(missing_df)
print(f"\nTotal columns with missing: {len(missing_df)}")
```

### Step 3: Target Distribution

Check if the classification target is balanced or imbalanced.

```python
target_col = 'Transported'  # Adjust for your dataset
print(f"\n--- Target Distribution ---")
print(df[target_col].value_counts())
print(f"\n--- Target Balance ---")
print(df[target_col].value_counts(normalize=True).round(3))
# Rule of thumb: < 70/30 split is considered balanced
```

### Step 4: Numerical Feature Distributions

Analyze each numerical feature's distribution and outliers.

```python
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical features ({len(numerical_cols)}): {numerical_cols}")

for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"\n{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
          f"std={df[col].std():.2f}, outliers={outliers}")
```

### Step 5: Categorical Feature Distributions

Check cardinality and value distributions for categorical features.

```python
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
# Remove target if it's boolean
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")

for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"\n{col} (unique={n_unique}):")
    if n_unique <= 20:
        print(df[col].value_counts())
    else:
        print(f"  Top 10: {df[col].value_counts().head(10).to_dict()}")
```

### Step 6: Feature-Target Relationships

Identify which features correlate most with the target.

```python
# Numerical features: compare means by target class
print("\n--- Mean by Target Class (Numerical) ---")
print(df.groupby(target_col)[numerical_cols].mean().round(2))

# Categorical features: cross-tabulation rates
print("\n--- Category vs Target Rates ---")
for col in categorical_cols:
    if df[col].nunique() <= 20:
        ct = pd.crosstab(df[col], df[target_col], normalize='index').round(3)
        print(f"\n{col}:")
        print(ct)
```

### Step 7: Correlation Matrix

Find multicollinearity between numerical features.

```python
corr = df[numerical_cols].corr()
print("\n--- Top Correlations ---")
# Get upper triangle pairs sorted by absolute correlation
pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for c1, c2, r in pairs[:10]:
    print(f"  {c1} <-> {c2}: {r:.3f}")
```

### Step 8: Save EDA Report

Write a structured markdown report summarizing all findings.

```python
report = f"""# EDA Report

## Dataset Overview
- Shape: {df.shape}
- Numerical features: {len(numerical_cols)}
- Categorical features: {len(categorical_cols)}

## Missing Values
{missing_df.to_markdown() if len(missing_df) > 0 else 'No missing values'}

## Target Distribution
{df[target_col].value_counts().to_markdown()}

## Key Findings
[Summarize the most important findings here]

## Recommendations for Feature Engineering
[List specific recommendations based on findings]
"""
with open('/home/gem/reports/eda_report.md', 'w') as f:
    f.write(report)
print("Report saved to /home/gem/reports/eda_report.md")
```

## Common Pitfalls

- **Don't** load the entire dataset into a single print statement — use `.head()` or summaries
- **Don't** skip missing value analysis — it's critical for feature engineering decisions
- **Don't** create matplotlib plots without saving to file (use `plt.savefig()`)
- **Don't** ignore the target distribution — imbalanced classes need special handling
- **Don't** forget to create the reports directory: `mkdir -p /home/gem/reports`
