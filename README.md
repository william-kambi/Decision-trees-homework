# ML Zoomcamp 2026: Homework 6 (Decision Trees, Random Forests, XGBoost)

Solutions to Homework 6 of [DataTalksClub's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (2026 cohort).

The goal is to predict car fuel efficiency (`fuel_efficiency_mpg`) with tree-based regression models and compare how hyperparameters affect validation RMSE.

## Dataset

[`car_fuel_efficiency_2026.csv`](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv) (about 10,000 rows).

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

## Setup

```bash
pip install pandas scikit-learn xgboost
python homework6.py
```

`homework6.py` runs every question and prints the results.

## Data preparation

- Missing values filled with `0`.
- 60/20/20 train/validation/test split, using the exact calls from the assignment (`random_state=1`).
- Features converted with `DictVectorizer(sparse=True)`, fitted on the training set only to avoid leakage.
- Original column names recovered from the one-hot features (for example `origin=USA` maps to `origin`).

## Results

| Q | Task | Answer |
|---|------|--------|
| 1 | Feature used by a depth-1 decision tree | `model_year` |
| 2 | Random forest (10 trees) validation RMSE | 1.837 (closest option; my run gave 1.833) |
| 3 | Best `n_estimators` from 10, 50, 100, 150 | 150 on my machine (100 and 150 are within 0.001) |
| 4 | Best `max_depth` by mean RMSE | 10 |
| 5 | Most important feature among the 4 | `vehicle_weight` |
| 6 | Best XGBoost `eta` | 0.1 |

## Reasoning

**Q1: Root split of a depth-1 tree.** A depth-1 tree has a single split. `tree_.feature[0]` gives the column it chose, and splitting the name on `=` maps one-hot columns back to the original feature. The tree splits on `model_year`.

**Q2: Random forest baseline.** A forest of 10 trees (`random_state=1`) gives a validation RMSE of about 1.83, meaning predictions are off by roughly 1.8 mpg on average.

**Q3: Number of trees.** RMSE drops sharply from 10 to 50 trees as averaging reduces variance, then flattens. The values at 100 and 150 trees differ by only about 0.001, so the winner can change with library version. The lowest value in my run was at 150 trees (1.769).

**Q4: Tree depth.** For each `max_depth` in [10, 15, 20, 25] I averaged RMSE over `n_estimators` in [10, 50, 100, 150], so one lucky tree count cannot decide the result. Depth 10 had the lowest mean RMSE (about 1.755), while 15, 20 and 25 were all around 1.78 to 1.79. Shallower trees generalized better, and deeper ones overfit.

**Q5: Feature importance.** `feature_importances_` sums each feature's impurity reduction across all splits and trees. Among the four candidates, `vehicle_weight` (about 0.20) clearly beat `horsepower`, `engine_displacement` and `acceleration`. This fits the physics, since heavier cars burn more fuel.

**Q6: XGBoost learning rate.** `eta` scales each tree's contribution. With `eta=0.3` the model learns quickly, reaches its best validation RMSE (1.732) around round 10, then overfits and ends at 1.826. With `eta=0.1` it learns more slowly, reaching a best of 1.712 around round 39 and ending at 1.725. Lower `eta` wins on both the final and the best score.

## Notes

- Small differences in RMSE (in the third decimal) can appear between machines because of scikit-learn and pandas versions, so the numbers above are from one environment.
- Q3 is the most sensitive to this, as the top two candidates are very close.


