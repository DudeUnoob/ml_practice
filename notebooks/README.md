# Practice notebooks

Jupyter notebooks and small side projects for classical machine learning with
scikit-learn, pandas, and matplotlib. These are separate from the NumPy
**deep learning from scratch** curriculum in the repository root.

## Setup

From the repository root:

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[notebooks]"
jupyter lab
```

For the Iris FastAPI example, also install the API extras:

```bash
python -m pip install -e ".[notebooks,api]"
```

## Projects

| Directory | Topic | Key techniques |
|-----------|-------|----------------|
| [iris_dataset](iris_dataset/) | Multi-class classification | EDA, train/test split, scatter plots |
| [logistic_regression_classification](logistic_regression_classification/) | Binary classification | Logistic regression, confusion matrix, GridSearchCV |
| [adult_classification](adult_classification/) | Tabular classification | Income prediction on UCI Adult dataset |
| [pollution_classification](pollution_classification/) | Tabular classification | Feature engineering on pollution data |
| [shopping_classification](shopping_classification/) | Customer segmentation | Shopping trends dataset |
| [lasso_regression](lasso_regression/) | Regression | OLS, Lasso, Ridge, Random Forest on wine quality |
| [time_series](time_series/) | Time series | Sea level pressure analysis (Darwin station) |
| [alice_comp](alice_comp/) | Competition template | Text + session features, logistic regression |
| [iris_api](iris_api/) | Model serving | Train Random Forest, expose via FastAPI |

Each project folder contains its notebook(s) and bundled CSV data where needed.
Open the notebook from its project directory so relative paths resolve correctly.

## Tips

- Start with **iris_dataset** if you want a gentle classification refresher.
- **lasso_regression** includes course material from [mlcourse.ai](https://mlcourse.ai).
- **alice_comp** keeps only the template notebook and session CSVs under `data/`.
  Submission artifacts are gitignored; regenerate them by running the notebook.
