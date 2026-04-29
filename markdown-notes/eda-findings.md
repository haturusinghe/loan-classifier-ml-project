| EDA Question | Why It Matters |
|---|---|
| What's the class balance of `loan_paid_back`? | If imbalanced → drives metric choice (ROC-AUC over accuracy) and `class_weight='balanced'` |
| Which columns have missing values and how many? | Determines imputation strategy per column |
| Which features are numeric vs categorical? | Drives your `ColumnTransformer` design |
| Are any numeric features heavily skewed? | May need log transform before scaling |
| Are any features correlated with each other? | Multicollinearity — may drop or combine |
| Are any features correlated with the target? | Identifies your strongest predictors |
| Are there any obvious data quality issues? | Wrong dtypes, nonsense values, duplicates |


# What's the class balance of `loan_paid_back`?
 
 **80/20 is a significant class imbalance**.

To put it in perspective: a completely dumb model that predicts `1` (loan repaid) for every single row would score **80% accuracy** without learning anything. That's exactly why accuracy is a dangerous metric here — it would make a useless model look great on paper.

Here's what this imbalance means for your specific decisions later:

**Evaluation metric** → Use **ROC-AUC** as your primary metric, not accuracy. ROC-AUC measures how well the model separates the two classes regardless of imbalance. F1-score on the minority class (`0`) is also useful.

**All 7 classifiers** → Pass `class_weight='balanced'` to every classifier that supports it (Logistic Regression, Decision Tree, Random Forest, SVM). This tells sklearn to penalise misclassifying the minority class more heavily.

**Kaggle submission** → The hidden test set likely has the same imbalance, so a model that learns both classes properly will outperform one that just predicts `1` most of the time.

# Which columns have missing values and how many?

**No missing values at all** — the dataset is completely clean on this front.

This is actually good news for two reasons:

- You don't need to make any imputation decisions, which removes a potential source of error
- You can skip the `SimpleImputer` step in your `ColumnTransformer` entirely, keeping your preprocessing pipeline simpler and easier to explain in the report

# `monthly_income` vs `annual_income` -- Are they same ?

 Look at these two rows side by side:

| annual_income | monthly_income |
|---|---|
| 30935.2 | 2577.94 |
| 35542.9 | 2961.91 |
| 93830.7 | 7819.22 |

 `monthly_income` and `annual_income` are **100% redundant** — one is literally calculated from the other. That would mean keeping both gives your model the same information twice, which has implications for your Q5 correlation analysis and your feature selection decisions.

That's a **confirmed perfect redundancy** — `monthly_income` is literally just `annual_income ÷ 12`, rounded to 2 decimal places. Every single row. Zero difference. [campus.datacamp](https://campus.datacamp.com/courses/preprocessing-for-machine-learning-in-python/selecting-features-for-modeling?ex=4)

What this means concretely:

- These two columns carry **identical information** — giving both to a model is like giving it the same feature twice with different units
- Keeping both can cause **multicollinearity** issues, especially for Logistic Regression and SVM where the math assumes features are relatively independent [codesignal](https://codesignal.com/learn/courses/intro-to-data-cleaning-and-preprocessing-with-titanic/lessons/understanding-and-handling-redundant-or-correlated-features-in-datasets)
- Tree-based models (Decision Tree, Random Forest) are less affected by this, but it's still noise

You now have two options to think about:

1. **Drop one** — keep `annual_income`, drop `monthly_income` (or vice versa). Simpler, cleaner
2. **Drop both, engineer a new one** — e.g. combine with `loan_amount` to create a `loan_to_income_ratio` feature, which might be more predictive than either raw income column alone

This is exactly the kind of domain-informed feature decision the assignment hint is pointing at when it says *"identifying or generating good features"*.


# Which features are numeric vs categorical?

### Numeric Columns (15)

These are columns pandas detected as numbers (`int` or `float`):

| Column | What it represents | Real-world range |
|---|---|---|
| `id` | Row identifier — not a feature | Just an index |
| `age` | Borrower's age in years | Probably 18–80 |
| `annual_income` | Yearly income in dollars | Varies widely |
| `monthly_income` | Monthly income in dollars | Likely annual ÷ 12 |
| `debt_to_income_ratio` | Total monthly debt payments ÷ gross monthly income | Healthy = below 0.36, risky = above 0.50   |
| `credit_score` | Creditworthiness score | 300–850; below 580 = poor, 670+ = good   |
| `loan_amount` | Amount borrowed in dollars | — |
| `interest_rate` | Annual interest rate (%) | — |
| `loan_term` | Loan duration in months | Typically 36 or 60   |
| `installment` | Monthly repayment amount | Derived from loan_amount, rate, term |
| `num_of_open_accounts` | Number of currently open credit accounts | — |
| `total_credit_limit` | Total credit available across all accounts | — |
| `current_balance` | Current outstanding debt balance | — |
| `delinquency_history` | Some score/count of past payment failures | — |
| `num_of_delinquencies` | Count of missed/late payments | — |

## Categorical Columns (5)

These are columns stored as text strings:

| Column | Categories | Nature |
|---|---|---|
| `gender` | Male, Female, Other | Nominal — no natural order |
| `marital_status` | Single, Married, Divorced, Widowed | Nominal — no natural order |
| `education_level` | High School, Bachelor's, Master's, PhD, Other | Has a natural order (but "Other" breaks it) |
| `employment_status` | Employed, Self-employed, Unemployed, Student, Retired | Nominal — categories are qualitatively different |
| `loan_purpose` | 8 categories (Debt consolidation, Car, Home, etc.) | Nominal — no natural order |

## Two Things Worth Noting Right Now

**`installment` is likely derived** — monthly installment is mathematically calculated from `loan_amount`, `interest_rate`, and `loan_term`. Similar to the `annual_income`/`monthly_income` situation. Worth checking in Q5.

**`loan_term`** — even though it's numeric, run `df['loan_term'].value_counts()` quickly. If it only has 2–3 values (like 36 and 60), it's effectively categorical despite being stored as a number.


**On the numeric columns:**
- `id` is in your numeric list — this must be dropped before training, it's just a row identifier with no predictive meaning
- `loan_term` is numeric but likely has only 2–3 distinct values (e.g. 36, 60 months) — run `df['loan_term'].value_counts()` quickly to confirm, because if so it behaves more like a categorical
- `delinquency_history` and `num_of_delinquencies` are both numeric and both about delinquency — likely correlated, Q5 will reveal this

**On the categorical columns:**
- All 5 are low-cardinality (3–8 unique values) — `OneHotEncoder` will handle these cleanly without creating too many columns
- `education_level` has a natural order (High School → Bachelor's → Master's → PhD) — you'll need to decide in Q6 whether it has a linear relationship with the target, which would make `OrdinalEncoder` more appropriate than `OneHotEncoder`
- `employment_status` categories like `Unemployed`, `Retired`, `Student` are qualitatively very different from `Employed` — this could be a strong predictor


# Are any numeric features heavily skewed?
Skewness measures how asymmetrical a feature's distribution is — basically, how much its data is "pulled" to one side compared to a perfect bell curve.

Imagine a histogram of your data. If it's perfectly symmetrical (bell-shaped), skewness = 0. If a few extreme values drag the tail to the right, skewness is positive. If extreme low values drag it left, it's negative. The mean always chases the tail — so in a right-skewed distribution, mean > median.

**Positive skew (long tail on the RIGHT):**
Most values are low/moderate, but a few extremely high values stretch the tail rightward.

**Negative skew (long tail on the LEFT):**
Most values are high/moderate, but a few extremely low values stretch the tail leftward.

### What the Magnitude Means [smartpls](https://smartpls.com/documentation/functionalities/excess-kurtosis-and-skewness/)

| Skewness value | Interpretation |
|---|---|
| Between -1 and +1 | Mild — distribution is roughly symmetric, acceptable |
| Between ±1 and ±2 | Moderate — noticeable asymmetry, worth noting |
| Beyond ±2 | Severe — heavily lopsided, likely has extreme outliers |

### Applied to Your Actual Data

**`current_balance = 2.849`** — Most borrowers have moderate balances (mean 24k), but a small number have extremely high balances up to 352,178. The majority cluster on the left, with a very long right tail. The Q7 stats confirm this: mean 24k but max 352k.

**`credit_score = -0.076`** — Nearly zero, almost perfectly symmetric. The mean (679) and median (680) are almost identical in Q7. This is why Pearson correlation was trustworthy for `credit_score` in Q6 — the distribution is clean.

**`interest_rate = 0.028`** — Also near zero, very symmetric. Confirms the Q6 correlation of 0.107 with the target is reliable.

**`annual_income = 2.343`** — Same story as `current_balance`. Most people earn 25k–55k but a few earn up to $400,000, dragging the tail to the right. This is exactly why `annual_income` showed near-zero correlation with the target in Q6 despite likely being financially meaningful — Pearson couldn't detect the relationship through all that skew. [codesignal](https://codesignal.com/learn/courses/feature-engineering-and-problem-handling-1/lessons/diagnosing-skewed-features-and-weak-correlations-in-your-dataset)

**`loan_term = 0.871`** — Moderate positive skew, but remember from Q7 this only has two values (36 and 60). The skew here just reflects that more borrowers choose 36-month terms than 60-month terms.

The practical takeaway: features with skewness above 2.0 (`current_balance`, `total_credit_limit`, `annual_income`, `monthly_income`) need `log1p` transformation before feeding into Logistic Regression and SVM, because those models are sensitive to scale and distribution shape.

#### Why Dropping Rows Is the Wrong Fix

**It introduces bias into your model**. The high-income borrowers ($200k–$400k annual income) are real people who will appear in `X_unknown`. If your model was trained without rows like them, it has never learned to handle that segment and will predict poorly for them during evaluation. [statisticsbyjim](https://statisticsbyjim.com/basics/remove-outliers/)

**It's data loss you can't afford.** You only have 18,000 rows. The class imbalance already means only 3,593 rows are class `0`. Dropping outlier rows disproportionately affects minority cases and makes the imbalance problem worse.

**The extreme values might be your most informative cases.** A borrower with $400k income behaves very differently from one with $30k income — that difference is signal, not noise. [towardsdatascience](https://towardsdatascience.com/skewness-and-kurtosis-with-outliers-f43167532c69/)

#### The Correct Mental Model

Dropping outliers fixes the *measurement tool* (Pearson correlation) by changing the *data*. That's backwards. You should fix the measurement tool instead:

| Problem | Wrong fix | Right fix |
|---|---|---|
| Skewed feature makes Pearson unreliable | Drop outlier rows | Apply `log1p` transform to the feature |
| Pearson can't detect non-linear relationships | Drop data | Use Spearman correlation instead |

#### Two Correct Approaches

**For EDA — switch to Spearman correlation for skewed features**: [linkedin](https://www.linkedin.com/advice/1/how-do-you-address-skewed-data-when-choosing-correlation-alrif)
```python
# Spearman works on ranks, not raw values — immune to outliers and skew
spearman_corr = df[num_cols].corrwith(df[TARGET], method='spearman').abs()
print(spearman_corr.sort_values(ascending=False).round(3))
```

Run this and compare the values to your Q6 Pearson results — the differences for `current_balance`, `total_credit_limit`, and `annual_income` will likely be significant, revealing the true relationship those features have with the target.

**For the preprocessing pipeline — transform, don't drop:**
```python
import numpy as np
df['annual_income_log'] = np.log1p(df['annual_income'])
df['current_balance_log'] = np.log1p(df['current_balance'])
df['total_credit_limit_log'] = np.log1p(df['total_credit_limit'])
```

`log1p` compresses the right tail, bringing extreme values closer to the bulk of the distribution without removing any rows. Your model still sees every borrower — but their income values are now on a scale that linear models can work with properly. [kaggle](https://www.kaggle.com/questions-and-answers/539888)


#### Why Blanket Application Is Wrong

`log1p` fixes one specific problem: **a long right tail pulling the distribution away from symmetry**. If a feature doesn't have that problem, applying `log1p` introduces a distortion that wasn't there before.

Look at your specific features:

| Feature | Skewness | Apply log1p? | Why |
|---|---|---|---|
| `current_balance` | 2.849 | ✅ Yes | Severely right-skewed |
| `total_credit_limit` | 2.533 | ✅ Yes | Severely right-skewed |
| `annual_income` | 2.343 | ✅ Yes | Severely right-skewed |
| `monthly_income` | 2.343 | ✅ Yes | Same as above |
| `credit_score` | **-0.076** | ❌ No | Almost perfectly symmetric — log1p would push it left-skewed |
| `interest_rate` | 0.028 | ❌ No | Already symmetric |
| `age` | 0.016 | ❌ No | Already symmetric |
| `loan_term` | 0.871 | ❌ No | Only two values (36/60) — transformation is meaningless |

`credit_score` is the clearest case — it has a skewness of -0.076, meaning it's already nearly perfect. Applying `log1p` to it would compress the lower scores and stretch the higher ones, **introducing** artificial skew where none existed. [apxml](https://apxml.com/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/log-transformation)

#### The Right Rule

Only apply `log1p` when skewness is **above 1.0** (or below -1.0 for left-skewed). That gives you a clean, justified threshold you can state explicitly in your report — which is exactly what Criterion 3 asks for.

# Are any features correlated with each other?


### Decision 1 — What to Drop Outright

These pairs have correlations so high that keeping both adds zero new information: [rohan-paul](https://www.rohan-paul.com/p/ml-interview-q-series-when-would-ff2)

| Pair | Correlation | Pipeline Decision |
|---|---|---|
| `annual_income` ↔ `monthly_income` | **1.000** | Drop `monthly_income` — keep `annual_income` |
| `loan_amount` ↔ `installment` | **0.944** | Drop `installment` — keep `loan_amount` |
| `delinquency_history` ↔ `num_of_delinquencies` | **0.903** | Keep one or engineer a combined feature |

Also drop `id` — it has near-zero correlation with everything, it's just a row index.

### Decision 2 — What to Handle Differently Per Model Type [linkedin](https://www.linkedin.com/posts/akhil-sharma-datasmartness_machinelearning-datascience-featureengineering-activity-7372905198861824000-hY3t)

This is the most important insight from the correlation matrix. The same correlated features affect models differently:

**Sensitive to multicollinearity:**
- Logistic Regression — unstable coefficients when features correlate [egarpor.github](https://egarpor.github.io/SSS2-UC3M/logreg-modsel.html)
- SVM — same issue, linear kernel especially

**Not affected by multicollinearity:**
- Decision Tree — splits on one feature at a time
- Random Forest, XGBoost — same, tree-based models don't care

This means you don't need one universal feature set. You can justify using slightly different feature sets per model family — which is a strong demonstration of understanding for Criterion 4 and the oral defence.

### Decision 3 — What to Engineer Into New Features

The income cluster correlation tells a specific story:

```
annual_income ↔ total_credit_limit:  0.887
annual_income ↔ current_balance:     0.654
total_credit_limit ↔ current_balance: 0.735
```

These three are correlated with each other but weak predictors of the target individually. The signal is locked inside the *relationship between* them, not in any single column. This points directly to: [towardsdatascience](https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6/)

```python
# Credit utilization rate — what % of available credit is being used
df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']

# Loan burden — loan size relative to income
df['loan_to_income'] = df['loan_amount'] / df['annual_income']
```

### Decision 4 — The credit_score ↔ interest_rate Pair Needs Special Thought

Correlation = **-0.568** — moderate but not extreme. Both have meaningful independent target correlation in Q6 (0.198 and 0.107 respectively). This means: [statisticalhorizons](https://statisticalhorizons.com/multicollinearity/)

- **For tree models**: keep both — they'll naturally pick whichever is more useful at each split
- **For LR/SVM**: worth considering keeping both but noting the collinearity exists in your report — it's not extreme enough to force dropping one

### How This Maps to Your `features.py`

```python
def engineer_features(df):
    df = df.copy()

    # Drop redundant columns
    df = df.drop(columns=['monthly_income', 'installment', 'id'])

    # Drop one from delinquency pair (or combine)
    df = df.drop(columns=['num_of_delinquencies'])  # keep delinquency_history

    # Engineer ratio features
    df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']
    df['loan_to_income'] = df['loan_amount'] / df['annual_income']

    return df
```

This function goes in `features.py` and gets called **before** the `ColumnTransformer` — it's a preprocessing step that operates on the raw dataframe before sklearn sees it.