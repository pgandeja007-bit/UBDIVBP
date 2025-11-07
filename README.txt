
STREAMLIT DASHBOARD PACKAGE for "Universal Bank - Personal Loan Propensity"

Contents (all files are in the root of the ZIP; no subfolders):
- app.py            : Streamlit app (main) — run with `streamlit run app.py`
- README.txt        : this file with instructions

How to deploy:
1. Create a new GitHub repository and upload BOTH files at repository root (no folders).
2. Connect the repo to Streamlit Cloud (https://streamlit.io/cloud) and deploy.
3. On the first run you can upload your UniversalBank.csv from the "Models" page or place the CSV at /mnt/data/UniversalBank.csv in a hosted environment if possible.
4. The app uses default packages: streamlit, pandas, numpy, scikit-learn, plotly. Streamlit Cloud provides these commonly used packages by default.

Features implemented in the app:
- Overview tab: 5 business-focused charts (Education x Family heatmap; Income distribution by loan acceptance; Income bin conversion rates; RandomForest feature importances; Age vs CCAvg scatter colored by loan status).
- Models tab: Train Decision Tree, Random Forest, Gradient Boosting with 5-fold CV, show metrics (CV mean/std, train/test accuracy, precision, recall, F1, AUC), ROC overlays, confusion matrices, and feature importances.
- Predict tab: Upload new dataset and predict 'Personal Loan' using trained models; download predictions CSV.
- Data & Dictionary tab: sample rows and the provided data dictionary.

Notes & limitations:
- The app trains models in-session; for production you may prefer pre-trained serialized models and a consistent preprocessing pipeline.
- The app expects the Universal Bank column names as described. Ensure 'Personal Loan' column exists when uploading training data.
