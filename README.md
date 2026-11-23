# Credit Risk Modelling: Application Scorecard and Challenger Models

A credit scorecard built on Lending Club loan data (2007–2014) to predict which loan applicants are likely to default, with a drift check and a machine learning challenger model for comparison.

## Overview

Lenders need a simple, explainable way to decide whom to approve. This project builds that decision tool in four steps:

1. **Prepare the data:** clean the Lending Club loan data and define the default (bad loan) target.
2. **Build the scorecard:** group each variable into bins using Weight of Evidence (WoE), rank variables by Information Value (IV), and fit a logistic regression on the WoE values.
3. **Score and set cutoffs:** convert the model output into a points-based score and choose approval cutoffs.
4. **Check and challenge:** monitor population drift with PSI, and benchmark a LightGBM challenger model explained with SHAP.

## Dataset

Lending Club loan data, 2007–2014 (`loan_data_2007_2014`), a public dataset of loans issued through the Lending Club platform.

## Methodology

| Step | What was done | Why it matters |
|---|---|---|
| Data preparation | Cleaned the loan data and defined good and bad loans | A clear target is needed before any model |
| WoE binning | Grouped each variable into bins and converted each bin to a risk number | Makes the model simple, stable and explainable |
| Information Value | Ranked variables by how well they separate good and bad borrowers | Keeps only useful variables |
| Logistic regression | Fitted the scorecard model on WoE values | Standard, transparent model used by lenders |
| Model evaluation | Measured AUROC, Gini and KS | Shows how well the model separates good and bad borrowers |
| Score and cutoffs | Converted model output into a points score with approval cutoffs | Turns a model into a decision rule |
| PSI monitoring | Compared the score distribution of new applicants with the training population | Flags when the model may need updating |
| LightGBM challenger | Built a second, more flexible model and explained it with SHAP | Tests whether a complex model beats the scorecard and shows which features drive predictions |

## Key Concepts

- **Scorecard:** a model that gives each applicant a score. A higher score means lower risk. Cutoffs on the score drive approve or reject decisions.
- **Weight of Evidence (WoE):** a number for each bin of a variable showing how risky that group is compared with the rest.
- **Information Value (IV):** a single number showing how useful a variable is for separating good and bad borrowers.
- **AUROC, Gini, KS:** three standard measures of how well a model ranks risky borrowers above safe ones.
- **PSI (Population Stability Index):** checks whether new applicants look different from the ones the model was built on.
- **Challenger model:** an alternative model compared against the main one to test whether it performs better.
- **SHAP:** a method that shows how much each feature pushed an individual prediction up or down.

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, Matplotlib, Jupyter Notebook.

## How to Run

1. Clone this repository.
2. Install the libraries:

   ```
   pip install pandas numpy scikit-learn lightgbm shap matplotlib jupyter
   ```

3. Place the Lending Club dataset (`loan_data_2007_2014`) in the same folder as the notebook.
4. Open the notebook in Jupyter and run the cells from top to bottom.

## Author

Aishi Ghosh, MA General Economics, Madras School of Economics.
