---
name: feature-engineering
description: Use this skill when preprocessing tabular data for machine learning. Handles missing values, categorical encoding, domain-specific feature extraction, and data leakage prevention. Always fit transformations on training data only.
---

# Feature Engineering for Tabular Classification

## Overview

Systematic feature engineering procedure for tabular classification datasets.
Takes raw data and produces clean, model-ready feature matrices (`X_train.csv`, `y_train.csv`, `X_test.csv`) with PassengerId preserved for submission.

## When to Use

- After EDA is complete and you understand the data
- When the orchestrator asks you to "preprocess", "engineer features", or "prepare data"
- Before any model training begins

## Best Practices

- **Fit only on train** — Never fit scalers, encoders, or imputers on test data
- **Transform both** — Apply the same fitted transformation to train AND test
- **Preserve PassengerId** — Never drop it before saving; it's needed for submission
- **Print outputs** — Always `print()` intermediate results so the agent captures them
- **Save processed data** — Write final matrices to `/home/gem/data/` as CSV files

## Process

### Step 1: Load Data and Separate Target

```python
import pandas as pd
import numpy as np

train = pd.read_csv('/home/gem/data/train.csv')
test = pd.read_csv('/home/gem/data/test.csv')

# Separate target before any processing
target_col = 'Transported'  # Adjust for your dataset
y_train = train[target_col].astype(int)  # Convert bool to 0/1
train_ids = train['PassengerId']
test_ids = test['PassengerId']

# Drop target and ID from feature set
train = train.drop(columns=[target_col, 'PassengerId'])
test = test.drop(columns=['PassengerId'])

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution:\n{y_train.value_counts()}")
```

### Step 2: Domain-Specific Feature Extraction

Extract structured information from composite columns before imputation.

**Example: Spaceship Titanic — Split Cabin into Deck/Num/Side**

```python
# Split composite columns (e.g., Cabin = "B/0/P" → Deck, Num, Side)
if 'Cabin' in train.columns:
    for df in [train, test]:
        df['Cabin_Deck'] = df['Cabin'].str.split('/').str[0]
        df['Cabin_Num'] = df['Cabin'].str.split('/').str[1].astype(float)
        df['Cabin_Side'] = df['Cabin'].str.split('/').str[2]
    train = train.drop(columns=['Cabin'])
    test = test.drop(columns=['Cabin'])
    print("Extracted Cabin → Cabin_Deck, Cabin_Num, Cabin_Side")

# Split PassengerId-like group features (e.g., "0001_01" → Group, PersonInGroup)
# Only if there's a pattern to exploit
```

### Step 3: Create Aggregate Features

Combine existing features to capture domain knowledge.

```python
# Example: Total spending feature (Spaceship Titanic)
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
existing_spending = [col for col in spending_cols if col in train.columns]
if existing_spending:
    for df in [train, test]:
        df['TotalSpending'] = df[existing_spending].sum(axis=1)
        df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    print(f"Created TotalSpending and HasSpending from {existing_spending}")

# Example: Age groups
if 'Age' in train.columns:
    for df in [train, test]:
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[0, 12, 18, 30, 50, 80],
            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
        )
    print("Created AgeGroup from Age")
```

### Step 4: Handle Missing Values

Strategy: Median for numerical, Mode for categorical. Always fit on train only.

```python
from sklearn.impute import SimpleImputer

numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")

# Impute numerical features — fit on TRAIN only
if numerical_cols:
    num_imputer = SimpleImputer(strategy='median')
    train[numerical_cols] = num_imputer.fit_transform(train[numerical_cols])  # FIT on train
    test[numerical_cols] = num_imputer.transform(test[numerical_cols])        # TRANSFORM test
    print(f"Imputed {len(numerical_cols)} numerical columns (median)")

# Impute categorical features — fit on TRAIN only
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train[categorical_cols] = cat_imputer.fit_transform(train[categorical_cols])  # FIT on train
    test[categorical_cols] = cat_imputer.transform(test[categorical_cols])        # TRANSFORM test
    print(f"Imputed {len(categorical_cols)} categorical columns (mode)")

# Verify no missing values remain
print(f"\nMissing after imputation — Train: {train.isnull().sum().sum()}, Test: {test.isnull().sum().sum()}")
```

### Step 5: Encode Categorical Features

Use LabelEncoder for ordinal or low-cardinality, OneHotEncoder for nominal features.

```python
from sklearn.preprocessing import LabelEncoder

# Label encode all categorical columns (simpler approach for tree-based models)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined unique values to handle unseen categories in test
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded {col}: {le.classes_[:5]}{'...' if len(le.classes_) > 5 else ''}")

print(f"\nFinal train shape: {train.shape}")
print(f"Final test shape: {test.shape}")
print(f"\nFeature dtypes:\n{train.dtypes}")
```

### Step 6: Save Processed Data

Save the feature matrices and target for model training.

```python
import os
os.makedirs('/home/gem/data', exist_ok=True)

# Save with PassengerId for reference
train_out = train.copy()
train_out.insert(0, 'PassengerId', train_ids.values)

test_out = test.copy()
test_out.insert(0, 'PassengerId', test_ids.values)

train_out.to_csv('/home/gem/data/X_train.csv', index=False)
y_train.to_csv('/home/gem/data/y_train.csv', index=False, header=['Transported'])
test_out.to_csv('/home/gem/data/X_test.csv', index=False)

print(f"\nSaved processed data:")
print(f"  X_train: {train_out.shape} → /home/gem/data/X_train.csv")
print(f"  y_train: {y_train.shape} → /home/gem/data/y_train.csv")
print(f"  X_test:  {test_out.shape} → /home/gem/data/X_test.csv")
print(f"\nFeature list: {list(train.columns)}")
```

## Anti-Patterns

- **Never** fit a scaler/encoder on test data — causes data leakage
- **Never** drop PassengerId before saving — it's needed for submission generation
- **Never** apply different transformations to train and test — causes train/test skew
- **Never** impute with mean on skewed distributions — use median instead
- **Never** one-hot encode high-cardinality features (>20 categories) — use label encoding
- **Never** create features that peek at test labels — that's data leakage

## Common Pitfalls

- **Forgetting to handle new categories** in test that don't exist in train → solution: fit LabelEncoder on combined train+test values
- **Dropping columns too early** → extract features first, then drop originals
- **Not printing shapes** after each step → makes debugging impossible
- **Creating features after imputation** when they should be created before → order matters
