# Spaceship Titanic — Exploratory Data Analysis Report

---

## 1. Dataset Overview

| Property | Train | Test |
|---|---|---|
| **Shape** | (8,693 × 14) | (4,277 × 13) |
| **Target** | `Transported` (bool) | — |
| **Memory** | 3,742 KB | 1,837 KB |

### Columns & Data Types

| Column | Type | Description |
|---|---|---|
| `PassengerId` | object | Unique ID, format `gggg_pp` (group + number) |
| `HomePlanet` | object | Origin planet (Earth, Europa, Mars) |
| `CryoSleep` | object | Whether in cryosleep (True/False) |
| `Cabin` | object | Cabin code, format `Deck/CabinNumber/Side` |
| `Destination` | object | Destination planet (3 values) |
| `Age` | float64 | Passenger age (0–79) |
| `VIP` | object | VIP status (True/False) |
| `RoomService` | float64 | Spending on room service |
| `FoodCourt` | float64 | Spending on food court |
| `ShoppingMall` | float64 | Spending on shopping mall |
| `Spa` | float64 | Spending on spa |
| `VRDeck` | float64 | Spending on VR deck |
| `Name` | object | Passenger first/last name |
| `Transported` | bool | **Target**: Whether transported to another dimension |

---

## 2. Missing Values

Missing values are **uniformly ~2%** across all columns — a manageable amount.

### Train Set Missing Values

| Column | Missing Count | Missing % |
|---|---|---|
| CryoSleep | 217 | 2.50% |
| ShoppingMall | 208 | 2.39% |
| VIP | 203 | 2.34% |
| HomePlanet | 201 | 2.31% |
| Name | 200 | 2.30% |
| Cabin | 199 | 2.29% |
| VRDeck | 188 | 2.16% |
| FoodCourt | 183 | 2.11% |
| Spa | 183 | 2.11% |
| Destination | 182 | 2.09% |
| RoomService | 181 | 2.08% |
| Age | 179 | 2.06% |

### Test Set Missing Values

Similar pattern: all columns have 1.9–2.5% missing. No columns are missing exclusively in train or test.

---

## 3. Target Distribution

| Transported | Count | Percentage |
|---|---|---|
| **True** | 4,378 | **50.36%** |
| **False** | 4,315 | **49.64%** |

**Verdict: The dataset is perfectly balanced.** No class weighting or resampling needed. The imbalance ratio is 1.015.

---

## 4. Feature Analysis

### 4.1 Categorical Features

#### HomePlanet
| Value | Count | % |
|---|---|---|
| Earth | 4,602 | 52.9% |
| Europa | 2,131 | 24.5% |
| Mars | 1,759 | 20.2% |

> Earth dominates. Europa and Mars are roughly 2:1.7 ratio.

#### CryoSleep
| Value | Count | % |
|---|---|---|
| False | 5,439 | 62.6% |
| True | 3,037 | 34.9% |

> ~35% of passengers were in cryosleep.

#### Destination
| Value | Count | % |
|---|---|---|
| TRAPPIST-1e | 5,915 | 68.0% |
| 55 Cancri e | 1,800 | 20.7% |
| PSO J318.5-22 | 796 | 9.2% |

> Heavily skewed toward TRAPPIST-1e.

#### VIP
| Value | Count | % |
|---|---|---|
| False | 8,291 | 95.4% |
| True | 199 | 2.3% |

> **Extreme imbalance**: only 2.3% VIP. Very low predictive power expected.

### 4.2 Numerical Features

#### Age
| Stat | Value |
|---|---|
| Mean / Median | 28.8 / 27.0 |
| Range | 0 – 79 |
| Skewness | 0.42 (mildly right-skewed) |
| Zeros | 178 (2.1%) |
| IQR Outliers | 77 (0.9%) |

> Age is well-distributed with a mild right skew. Most passengers are 19–38 years old. Children (0–12) are a meaningful segment (7.4%).

#### Spending Features (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck)

All spending features share a **distinctive zero-inflated, highly right-skewed pattern**:

| Feature | Zeros % | Mean | Median | Skewness | IQR Outliers % |
|---|---|---|---|---|---|
| RoomService | 65.5% | 224.7 | 0 | 6.3 | 21.9% |
| FoodCourt | 64.1% | 458.1 | 0 | 7.1 | 21.4% |
| ShoppingMall | 65.8% | 173.7 | 0 | 12.6 | 21.6% |
| Spa | 62.6% | 311.1 | 0 | 7.6 | 21.0% |
| VRDeck | 64.6% | 304.9 | 0 | 7.8 | 21.3% |

> **Key insight**: ~63–66% of passengers have **zero spending** in each category. The maximum values are extreme (14K–30K). These are **not true outliers** — they represent genuine high-spenders. Log transformation recommended for modeling.

---

## 5. Special Feature Analysis

### 5.1 Cabin Parsing (Deck / CabinNumber / Side)

#### Deck Distribution
| Deck | Count | % | Transported Rate |
|---|---|---|---|
| F | 2,794 | 32.9% | 44.0% |
| G | 2,559 | 30.1% | 51.6% |
| E | 876 | 10.3% | 35.7% |
| B | 779 | 9.2% | **73.4%** |
| C | 747 | 8.8% | **68.0%** |
| D | 478 | 5.6% | 43.3% |
| A | 256 | 3.0% | 49.6% |
| T | 5 | 0.1% | 20.0% (too small) |

> **Critical finding**: Decks B and C have dramatically higher Transported rates (73% and 68%). Decks E and D have the lowest (36% and 43%).

#### Deck ↔ HomePlanet Mapping (Deterministic)
| Deck | Earth | Europa | Mars |
|---|---|---|---|
| A | 0% | **100%** | 0% |
| B | 0% | **100%** | 0% |
| C | 0% | **100%** | 0% |
| D | 0% | 40% | **60%** |
| E | 46% | 15% | 39% |
| F | **59%** | 0% | 41% |
| G | **100%** | 0% | 0% |
| T | 0% | **100%** | 0% |

> This is a **strong deterministic relationship** — Deck can fill missing HomePlanet and vice versa.

#### Side Distribution
| Side | Count | Transported Rate |
|---|---|---|
| P (Port) | 4,206 | 45.1% |
| S (Starboard) | 4,288 | **55.5%** |

> Starboard side passengers are 10pp more likely to be Transported.

#### CabinNumber (binned)
| Bin | Transported Rate |
|---|---|
| 0–300 | 54.0% |
| 300–600 | 41.2% |
| 600–900 | 54.3% |
| 900–1200 | **60.4%** |
| 1200–1500 | 41.9% |
| 1500–1900 | 41.5% |

> Non-linear relationship — some cabin number ranges are more predictive than others.

### 5.2 PassengerId Groups

PassengerId format `gggg_pp` allows extraction of **group number** and **group size**.

| GroupSize | Count | % | Transported Rate |
|---|---|---|---|
| 1 (Solo) | 4,805 | 55.3% | **45.2%** |
| 2 | 1,682 | 19.4% | 53.8% |
| 3 | 1,020 | 11.7% | **59.3%** |
| 4 | 412 | 4.7% | **64.1%** |
| 5 | 265 | 3.0% | 59.2% |
| 6 | 174 | 2.0% | 61.5% |
| 7 | 231 | 2.7% | 54.1% |
| 8 | 104 | 1.2% | 39.4% |

> **Solo travelers are significantly less likely to be Transported** (45.2%) vs group travelers (56.7%). Group sizes 3–6 show the highest rates.

#### Group-Level Consistency
- **HomePlanet is 100% consistent within groups** — all group members share the same HomePlanet
- **Cabin is mostly shared** — 93% of groups share a single cabin (some groups span multiple cabins)
- These relationships enable **group-based imputation** for missing values

### 5.3 Spending Analysis

#### TotalSpend
| Stat | Value |
|---|---|
| Mean | 1,440.87 |
| Median | 716.00 |
| Skewness | 4.42 |
| Zero-spend passengers | **3,653 (42.0%)** |
| All-spend-zero (NoSpend) | **3,247 (37.4%)** |

#### NoSpend vs Transported — THE key relationship
| NoSpend | Count | Transported Rate |
|---|---|---|
| **Yes (all zeros)** | 3,247 | **78.4%** |
| No (has some spend) | 5,446 | **33.7%** |

> This is one of the **strongest predictors** in the dataset. Passengers who spent nothing are 2.3× more likely to be Transported.

#### CryoSleep ↔ Spending (Perfect Relationship)
- CryoSleep = True → **100% have zero spending** (confirmed: 0 out of 3,037 cryosleepers have any spend > 0)
- This is a **deterministic rule**: CryoSleep passengers cannot spend
- CryoSleep has 347 missing rows where spending = 0 → likely CryoSleep = True

#### Spending Category Split by Transported

| Spending Feature | No Spend → Transported Rate | Has Spend → Transported Rate |
|---|---|---|
| RoomService | 62.8% | **26.0%** |
| FoodCourt | 58.9% | 34.5% |
| ShoppingMall | 59.7% | **31.7%** |
| Spa | 63.4% | **27.7%** |
| VRDeck | 62.4% | **27.5%** |

> **RoomService, Spa, and VRDeck** show the strongest spending → Not Transported relationship. Spending on FoodCourt is the weakest predictor among spending features.

#### TotalSpend Binned
| Bin | Transported Rate |
|---|---|
| **0 (no spend)** | **78.6%** |
| 1–100 | 28.1% |
| 101–500 | 26.0% |
| 501–1,000 | 29.4% |
| 1,001–2,000 | 26.9% |
| 2,001–5,000 | 32.9% |
| 5,001+ | 32.2% |

> The zero/non-zero divide is far more important than the actual spending amount.

---

## 6. Feature-Target Relationships — Correlation Analysis

### Pearson Correlation with Transported

| Feature | Correlation | Direction |
|---|---|---|
| **CryoSleep** | **+0.469** | Strong positive |
| **NoSpend** | **+0.433** | Strong positive |
| RoomService | -0.245 | Moderate negative |
| Spa | -0.221 | Moderate negative |
| VRDeck | -0.207 | Moderate negative |
| Deck | +0.202 | Moderate positive |
| TotalSpend | -0.200 | Moderate negative |
| HomePlanet | +0.195 | Moderate positive |
| Solo | -0.114 | Weak negative |
| Destination | +0.110 | Weak positive |
| Side | +0.104 | Weak positive |
| GroupSize | +0.083 | Weak positive |
| Age | -0.075 | Weak negative |
| FoodCourt | +0.047 | Very weak |
| VIP | -0.038 | Very weak |
| ShoppingMall | +0.010 | Negligible |

### Multicollinearity Among Spending Features
- Spending features have **low inter-correlation** (mostly -0.02 to +0.23)
- Exception: FoodCourt correlates moderately with Spa (+0.22) and VRDeck (+0.23)
- TotalSpend is naturally correlated with individual features (up to 0.75 with FoodCourt)
- **Recommendation**: Use individual spending features + binary spend flags rather than just TotalSpend

---

## 7. Top Predictive Features (Ranked)

| Rank | Feature | Type | Key Insight | Importance |
|---|---|---|---|---|
| 1 | **CryoSleep** | Binary | 81.8% vs 32.9% Transported rate | ★★★★★ |
| 2 | **NoSpend** (derived) | Binary | 78.4% vs 33.7% Transported rate | ★★★★★ |
| 3 | **Deck** (B/C vs others) | Categorical | 73%/68% vs 35–52% Transported rate | ★★★★☆ |
| 4 | **HomePlanet** (Europa) | Categorical | 65.9% Europa vs 42.4% Earth | ★★★★☆ |
| 5 | **RoomService spend flag** | Binary | 26.0% vs 62.8% Transported rate | ★★★★☆ |
| 6 | **Spa spend flag** | Binary | 27.7% vs 63.4% Transported rate | ★★★★☆ |
| 7 | **GroupSize** (>1) | Ordinal | 56.7% group vs 45.2% solo Transported rate | ★★★☆☆ |
| 8 | **Destination** (55 Cancri e) | Categorical | 61.0% vs 47.1% TRAPPIST-1e | ★★★☆☆ |
| 9 | **Side** (S vs P) | Binary | 55.5% vs 45.1% Transported rate | ★★☆☆☆ |
| 10 | **Age** (0–12 children) | Numerical | 66.9% for age 0–12 vs ~48% for adults | ★★☆☆☆ |
| 11 | **VIP** | Binary | 38.2% vs 50.6% — very weak predictor | ★☆☆☆☆ |

---

## 8. Data Quality Summary

| Issue | Severity | Details |
|---|---|---|
| Missing values (~2% all columns) | **Low** | Uniformly distributed, easy to impute |
| Zero-inflated spending | **Structural** | 62–66% zeros per category; use binary flags |
| Heavy spending skewness | **Moderate** | Skewness 6–13; log-transform for modeling |
| VIP extreme imbalance | **Low** | Only 2.3% VIP; minimal predictive value |
| Cabin format complexity | **Low** | Parse into Deck/CabinNumber/Side |
| Deck T tiny sample | **Low** | Only 5 passengers; consider merging with Deck A |
| Age zeros (178) | **Low** | Likely infants (age 0); legitimate |

---

## 9. Recommendations for Feature Engineering

### High-Priority Transformations
1. **Parse Cabin** → `Deck`, `CabinNumber`, `Side` (drop raw Cabin)
2. **Parse PassengerId** → `Group`, `GroupSize`, `Solo` flag
3. **Create spending flags**: `HasRoomService`, `HasFoodCourt`, etc. (binary 0/1)
4. **Create `NoSpend`** flag (all spending = 0) — one of the strongest predictors
5. **Create `TotalSpend`** and consider **log-transformed** version
6. **Create `Child` flag** (Age < 13) — children have 67% Transported rate
7. **Group-based imputation**: Use group membership to fill HomePlanet, Deck, and Side

### Imputation Strategy
- **CryoSleep**: Use NoSpend as a guide — if all spending = 0, likely CryoSleep = True
- **HomePlanet**: Use Deck (deterministic mapping for A/B/C/G/T), or group members' HomePlanet
- **Cabin**: Use group members' cabin (93% of groups share a cabin)
- **Spending features**: Impute 0 for CryoSleep = True passengers; median for others
- **Age**: Median imputation (~27) or group-based imputation
- **VIP**: Impute False (95.4% are non-VIP)

### Encoding Strategy
- **CryoSleep, VIP, NoSpend, Solo, Child**: Binary (0/1)
- **Deck**: Ordinal encoding (grouped: B/C high, G medium, A/D/E/F/T low) or one-hot
- **HomePlanet**: One-hot encoding (3 categories)
- **Destination**: One-hot encoding (3 categories)
- **Side**: Binary (P=0, S=1)
- **Spending features**: Keep raw + log-transformed + binary flags
- **GroupSize**: Keep as numeric or binned

### Features to Drop
- `Name` — unique per passenger (8,473/8,493 unique); surname could be used for group detection but Group already captures this
- `PassengerId` — after extracting Group information
- `Cabin` — after parsing into Deck/CabinNumber/Side
- `ShoppingMall` — weakest predictor among spending features; correlation with target near zero

### Interaction Features to Consider
- `CryoSleep × NoSpend` — highly correlated (CryoSleep → NoSpend)
- `HomePlanet × Deck` — deterministic relationship, may cause leakage if both used
- `GroupSize × NoSpend` — large groups may be more likely to have zero spend
- `Age × CryoSleep` — children in cryosleep may have distinct patterns

---

## 10. Key Insights Summary

1. **CryoSleep is the #1 predictor** — 82% of cryosleepers were Transported vs only 33% of non-cryosleepers. This single feature explains a massive portion of the variance.

2. **Zero spending = high Transported probability (78%)** — The binary NoSpend flag is nearly as powerful as CryoSleep and captures overlapping information.

3. **CryoSleep and spending are deterministically linked** — 100% of CryoSleep passengers have zero spending. This means these features carry the same signal. Missing CryoSleep can be inferred from spending = 0.

4. **Deck matters enormously** — Passengers on Decks B and C (exclusively Europa) have 73% and 68% Transported rates, while Decks D and E have only 36–43%.

5. **HomePlanet is a strong proxy** — Europa passengers (65.9%) are far more likely to be Transported than Earth passengers (42.4%), largely mediated by Deck assignment.

6. **Group travel increases Transported probability** — Solo travelers (45%) are significantly less likely than group travelers (57%), with the effect strongest for group sizes 3–6.

7. **Children (0–12) are more likely to be Transported** (67%) than adults (~48%), making age binning valuable.

8. **VIP status is nearly useless** — Only 2.3% VIP, and they're slightly LESS likely to be Transported. Consider dropping.

9. **The spending amount matters less than spending at all** — Among spenders, Transported rate is consistently 27–35% regardless of spending level. The zero/non-zero divide is what matters.

10. **Among spending categories, RoomService, Spa, and VRDeck are the strongest negative predictors** — spending on these luxury amenities correlates with NOT being Transported.
