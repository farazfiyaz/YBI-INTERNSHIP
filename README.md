# Big Sales Prediction — Random Forest Regressor

A regression project that predicts `Item_Outlet_Sales` for a retail-store dataset of 14,204 transactions across 10 outlets, using a Random Forest Regressor on engineered categorical and standardized numeric features. Completed as part of the **YBI Foundation Data Science & Machine Learning Internship** (June 2023).

---

## Dataset

- **Source:** [YBI Foundation — Big Sales Data](https://github.com/YBI-Foundation/Dataset/raw/main/Big%20Sales%20Data.csv)
- **Rows:** 14,204
- **Columns:** 12 (1 target, 10 features after dropping `Item_Identifier`)
- **Target:** `Item_Outlet_Sales` (continuous)

## What I did

1. **Missing-value imputation** — `Item_Weight` had 2,389 nulls; imputed by group mean within each `Item_Type`.
2. **Categorical cleanup** — collapsed inconsistent labels in `Item_Fat_Content` (`LF`, `low fat` → `Low Fat`; `reg` → `Regular`).
3. **Manual encoding** — mapped `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type` to integers. `Item_Type` was collapsed to 3 buckets (food, non-food, other).
4. **Feature scaling** — `StandardScaler` on the 4 continuous features (`Item_Weight`, `Item_Visibility`, `Item_MRP`, `Outlet_Establishment_Year`).
5. **Train/test split** — 90/10 with `random_state=2529` (12,783 train / 1,421 test).
6. **Model** — `RandomForestRegressor(random_state=2529)` with default hyperparameters.

## Results

| Metric | Value |
|---|---|
| MSE  | ~1,613,965 |
| MAE  | ~828 |
| R²   | **~0.58** |

A 10/90 test split on a noisy retail dataset with default RF hyperparameters — R² of 0.58 is a reasonable baseline. Improvements would come from hyperparameter tuning (`n_estimators`, `max_depth`), one-hot rather than label encoding for the high-cardinality outlet IDs, and target log-transform for the right-skewed sales distribution.

## Tech stack

Python · pandas · NumPy · scikit-learn · seaborn · Matplotlib

## Repository contents

```
YBI-INTERNSHIP/
├── internshipproject1.ipynb   # Full notebook — EDA, preprocessing, model, evaluation
├── README.md
├── LICENSE
└── .gitignore
```

## Reproduce

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
jupyter notebook internshipproject1.ipynb
```

The dataset is fetched directly from the YBI Foundation GitHub URL inside the notebook — no local download needed.

## Internship

**YBI Foundation — Data Science & Machine Learning Internship** (June 2023). YBI Foundation is an AICTE-recognized non-profit educational organization based in Delhi, India.

## License

MIT — see [LICENSE](LICENSE). Dataset credit: YBI Foundation.
