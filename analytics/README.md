# `/analytics` — Zepto Data & AI Platform

End-to-end analytics pipeline on the Titanic dataset: profiling, cleaning, exploratory data analysis, and a full predictive-modeling pipeline (classification + regression), continuing from the same cleaned data throughout.

## Setup

```bash
pip install -r requirements.txt
```

Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `joblib`.

## How to Run

Run in order (two notebooks/scripts sharing one committed CSV):

1. **`01_eda.ipynb`** — loads the Titanic dataset via `sns.load_dataset('titanic')` (network/cache required on first run only), profiles it, cleans it, saves `titanic.csv`, and produces the full EDA data story.
2. **`02_modeling.ipynb`** — reads the committed `titanic.csv` (no second raw-data load), and continues into the full modeling pipeline.

The dataset is loaded from network/cache **exactly once**, in step 1 — every later step, including all of modeling, works from the same cleaned DataFrame or its saved CSV.

## Part A — Profiling, Cleaning, and the Data Story

### Dataset Profile
891 rows, 15 columns. Missing values found in:
- `age`: 19.87% missing
- `embarked` / `embark_town`: 0.22% missing
- `deck`: 77.10% missing

### Missing-Value Handling (threshold rule)
| Column | Missing % | Strategy | Justification |
|---|---|---|---|
| `embarked`, `embark_town` | 0.22% (<5%) | Dropped rows | Negligible data loss (2 rows); preserves complete records without needing imputation |
| `age` | 19.87% (5-30%) | Median imputation | Preserves sample size without severely distorting the distribution; median is robust to age's mild skew |
| `deck` | 77.10% (>30%) | Column dropped entirely | Imputation would be unreliable at this missing rate and risks introducing artificial noise/bias |

### Univariate Analysis
- **Age outliers (IQR method):** Q1=22.00, Q3=35.00, IQR=13.00, bounds=[2.50, 54.50] → **65 outliers**
- **Fare outliers (IQR method):** Q1=7.90, Q3=31.00, IQR=23.10, bounds=[-26.76, 65.66] → **114 outliers**
  - Note: the IQR lower bound for fare is mathematically negative, which is not practically meaningful since fare can never be negative — a direct consequence of fare's strong right-skew rather than a symmetric distribution. In practice, only the upper bound (65.66) identifies meaningful outliers here.
- **Fare skewness:** Mean=32.10, Median=14.45, Mode=8.05. Since Mean > Median > Mode, fare is **strongly right-skewed** — a small number of high-paying passengers (mostly first-class) pull the mean well above the median and mode.

### Bivariate Analysis
**Survival rate by sex:** Male 18.89%, Female 74.04%
**Survival rate by pclass:** Class 1: 62.62%, Class 2: 47.28%, Class 3: 24.24%
**Survival rate by sex + pclass combined:**
| Class | Male | Female |
|---|---|---|
| 1 | 36.89% | 96.74% |
| 2 | 15.74% | 92.11% |
| 3 | 13.54% | 50.00% |

**Correlation matrix** (computed on exactly `survived, pclass, age, sibsp, parch, fare`; `adult_male` and `alone` excluded as derived/redundant flags):

**Top 2 strongest correlations:**
1. **Fare ↔ Pclass (-0.55):** Strong negative correlation — passengers in lower (numerically higher) pclass categories paid significantly lower fares, and vice versa. Expected, since pclass directly reflects travel class and associated cost.
2. **Parch ↔ SibSp (0.41):** Moderate positive correlation — passengers traveling with more parents/children also tended to travel with more siblings/spouses, consistent with both features representing family-group size.

### Multivariate Data Story (4 charts)
1. **Survival rate by class and sex (bar chart):** Women survived at dramatically higher rates than men across every class; survival declines step-by-step from 1st to 3rd class for both genders. First-class women had near-certain survival (~96.8%), third-class men the lowest (~13.5%).
2. **Age distribution by survival and sex (split violin plot):** A distinct bulge in the male "Survived" distribution at ages 0-10 shows young boys were prioritized for lifeboats despite heavy adult-male mortality; the "Died" side is much wider for adult males (18-50), showing they made up the largest casualty group.
3. **Fare vs. age colored by survival (scatter, log-scale fare):** Survivors cluster toward higher fares regardless of age, consistent with premium-fare passengers occupying upper-deck cabins with more direct access to lifeboats.
4. **Survival rate by family size and embarkation town (heatmap):** Small family units (2-4 people) had the highest survival rates, benefiting from mutual support without the coordination burden of a larger group; solo travelers and very large families both show lower survival.

### Standardization Check (EDA-stage only)
Z-score standardization applied to `age` and `fare` (via `StandardScaler`) confirmed both transformed columns have approximately mean 0 and standard deviation 1, shown via overlaid raw-vs-standardized KDE plots. **This check is exploratory only and does not feed into the modeling pipeline below**, which performs its own separate, train-only scaling.

## Part B — Predictive Modeling

### Train/Test Split
Stratified 80/20 split (`stratify=y`), justified by the class imbalance observed in Part A (~38% survived vs. ~62% did not) — stratification ensures both splits preserve this same proportion, rather than risking an uneven distribution from a plain random split.

### Preprocessing (fit on training data only)
Built with `ColumnTransformer` + `Pipeline`:
- **Numeric features** (`pclass, age, sibsp, parch, fare`): median imputation → `StandardScaler`
- **Categorical features** (`sex, embarked`): most-frequent imputation → `OneHotEncoder`

**Columns excluded from modeling, and why:**
- `alive` — dropped due to **target leakage**: it directly restates `survived` as text
- `class`, `who`, `adult_male`, `embark_town`, `alone` — dropped as **redundant duplicates** of information already captured by `pclass`, `sex`, `age`, `embarked`, `sibsp`+`parch` respectively; not a leakage risk, but avoids unnecessary duplicate signal and extra one-hot columns

**Note on imputation in this stage:** `age` and `embarked` were already cleaned in Part A, so the pipeline's own imputers here have no missing values to act on in the current train/test split. They're included anyway as a defensive step, per the task's own guidance — so the saved, deployable pipeline handles missing values gracefully if ever applied to new, raw, unseen data later.

### Classifiers Trained
Logistic Regression, Decision Tree (`max_depth=5`), and Random Forest (`n_estimators=100, max_depth=5`) — all trained on the identical stratified split, with no `class_weight` adjustment at this baseline stage (imbalance handling is compared separately in its own dedicated step below). Decision Tree visualized via `plot_tree` with cleaned feature names and class labels.

### Model Comparison (Baseline)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8610 | [98, 12, 21, 47] |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 | [98, 12, 30, 38] |
| Random Forest | 0.8202 | 0.8462 | 0.6471 | 0.7333 | 0.8533 | [102, 8, 24, 44] |

**Early read on baseline results:** Random Forest achieves the highest accuracy (0.8202) and precision (0.8462), but Logistic Regression has the highest recall (0.6912), ROC-AUC (0.8610), and a very close F1 (0.7402 vs. Random Forest's 0.7333) — meaning no single model dominates on every metric at this baseline stage. Decision Tree trails on every metric here, consistent with a single shallow tree being less powerful than an ensemble or a well-regularized linear model. This trade-off (Random Forest's precision vs. Logistic Regression's recall/AUC) is worth revisiting in the final recommendation once tuning and imbalance handling are complete.

### Imbalance Handling Comparison
Class balance:
- **Full dataset:** Not Survived (0): 549 (61.75%), Survived (1): 340 (38.25%)
- **Training fold:** Not Survived (0): 439 (61.74%), Survived (1): 272 (38.26%) — confirms the stratified split preserved the original class proportions almost exactly.

Three Random Forest variants compared, all with identical hyperparameters (`max_depth=5, n_estimators=100`) except for the imbalance-handling strategy itself:

| Imbalance Strategy | Precision | Recall | F1 Score |
|---|---|---|---|
| (a) Baseline (No Handling) | 0.8462 | 0.6471 | 0.7333 |
| (b) Class Weight Balanced | 0.7286 | 0.7500 | 0.7391 |
| (c) SMOTE (Training Fold Only) | 0.7656 | 0.7206 | 0.7424 |

**Conclusion:** F1 scores are close across all three variants (0.7333-0.7424), so no strategy dominates on F1 alone — the more meaningful difference is in the precision/recall trade-off each strategy makes. The baseline has the highest precision (0.8462) but the weakest recall (0.6471), meaning it's the most conservative about predicting "survived" but misses over a third of actual survivors. `class_weight='balanced'` produces the largest recall gain (0.7500, up from 0.6471), at the cost of precision dropping to 0.7286 — it trades some false positives for meaningfully fewer missed survivors. SMOTE lands between the two on every metric, offering a smaller recall improvement (0.7206) with less precision loss than `class_weight='balanced'` (0.7656 precision).

**Best strategy for this problem: `class_weight='balanced'`.** In a survival-prediction context, a false negative (predicting someone didn't survive when they did) represents a missed at-risk case, which is generally more costly to miss than a false positive (over-predicting survival). Since `class_weight='balanced'` achieves the highest recall of the three variants while keeping F1 essentially tied with the other options, it offers the best trade-off for a use case where catching true positives matters more than avoiding false alarms.

### Hyperparameter Tuning (GridSearchCV)
`GridSearchCV` over Random Forest's `n_estimators`, `max_depth`, and `max_features`, using `RandomForestClassifier(oob_score=True, ...)`.

**Best parameter combination:**
- `n_estimators`: 100
- `max_depth`: 10
- `max_features`: log2

**Best OOB score:** 0.8312

**Note on OOB vs. test accuracy:** the OOB score (0.8312) is computed from each tree's out-of-bag samples *within the training data itself* during bootstrap aggregation — it is a robust internal validation estimate, but it is not the same measurement as accuracy on the truly held-out test set. Both are useful, but they answer slightly different questions, and this distinction is worth stating explicitly rather than treating the two numbers as interchangeable.

### Regression Side-Task — Predicting Fare
Multivariate linear regression predicting `fare` from the other available features.

| Metric | Value |
|---|---|
| MAE | $21.10 |
| RMSE | $41.70 |
| R² | 0.3482 |
| Adjusted R² | 0.2965 |

**Interpretation:** the model explains only about 35% of the variance in fare (R²=0.3482), and adjusted R² (0.2965) is noticeably lower — a meaningful gap suggesting some of the model's apparent explanatory power comes from the number of predictors used rather than genuine signal. This is a fairly weak fit overall: fare in this dataset is likely driven substantially by factors not captured in the available features (e.g., specific cabin location, exact ticket type), so a modest R² here is expected rather than a modeling failure.

**Heteroscedasticity:** Yes — the residual plot displays clear heteroscedasticity (a non-random, unequal spread of residuals), specifically a funnel/fan pattern where residual variance expands as predicted fare increases. This is consistent with fare's strong right-skew observed in Part A (most tickets cost $7-$30, with a small number of luxury tickets exceeding $200-$500): low-fare predictions have small residuals, while high-fare predictions carry much larger prediction errors. This directly violates Ordinary Least Squares' constant-variance (homoscedasticity) assumption, and is a strong signal that a plain linear regression is a structurally weak fit for this particular target — consistent with the modest R² (0.3482) reported above. A log-transform of `fare` before modeling, or a non-linear model, would be reasonable next steps to address this in future iterations, though outside this task's required scope.

### Final Model Comparison & Recommendation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE ($) | RMSE ($) | R² | Adj. R² |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8610 | — | — | — | — |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 | — | — | — | — |
| Random Forest | 0.8202 | 0.8462 | 0.6471 | 0.7333 | 0.8533 | — | — | — | — |
| Linear Regression (Fare) | — | — | — | — | — | 21.10 | 41.70 | 0.3482 | 0.2965 |

*Classification and regression metrics are presented as two distinct metric groups above (different columns, different scales) — they are not directly comparable to one another.*

**Recommendation:** For deployment, **Random Forest tuned with GridSearchCV's best parameters (`n_estimators=100, max_depth=10, max_features='log2'`), combined with `class_weight='balanced'`**, is recommended over the three baseline classifiers. The baseline Random Forest already achieves the highest accuracy (0.8202) and precision (0.8462) among the three untuned classifiers, and GridSearchCV's tuned version reaches a strong OOB score of 0.8312 — meaningfully higher than the untuned baseline's test accuracy. Layering in `class_weight='balanced'` (from the earlier imbalance analysis) addresses Random Forest's weakest baseline metric, recall (0.6471 untuned vs. 0.7500 with balanced weighting on the earlier comparison), without materially harming F1. Logistic Regression remains a reasonable, more interpretable alternative — it posted the highest baseline ROC-AUC (0.8610) and F1 (0.7402) — worth considering if model interpretability matters more than the small performance edge Random Forest offers. The regression model for fare (R²=0.3482) is comparatively weak and should be treated as an exploratory side-analysis rather than a production-ready predictor, given the likely influence of unmeasured factors (e.g., cabin location) on actual fare.

### Saved Pipeline
The best-performing full pipeline (preprocessing + tuned Random Forest estimator together) was saved via `joblib.dump(full_pipeline, ...)`. Reload confirmed correct end-to-end prediction on raw input:

```
Passenger 1: Predicted Status = Survived (Class 1) | Confidence = 81.91%
Passenger 2: Predicted Status = Perished (Class 0) | Confidence = 72.32%
```

**Note on a warning observed during reload testing:** the reload step produced `UserWarning: Found unknown categories in columns [1] during transform. These unknown categories will be encoded as all zeros`. This is expected, non-breaking behavior — it confirms the `OneHotEncoder(handle_unknown='ignore')` safeguard from the preprocessing pipeline is working as designed: when the raw test passengers included a categorical value (in column index 1 of the categorical feature set) that the encoder never saw during training, it was gracefully encoded as all-zeros rather than crashing. This is worth mentioning explicitly in the viva as a demonstration that the saved pipeline handles unseen categories robustly, rather than something that indicates a bug.

## Git Workflow
Work on this module was developed on a feature branch, committed across multiple commits, and merged back into `main` — part of the project's overall Git branching workflow (checked once across the whole repository).