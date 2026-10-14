# Nedbank demand classification - Streamlit

This is a working Streamlit implementation of Question 7 using the exact saved
optimised classifier from the assignment. No model retraining is needed.

## Run locally

Use Python 3.12. Extract this folder, open a terminal inside it, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS alternative: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501. This is a local address, not a public deployment.

## Deploy to Streamlit Community Cloud

Public deployment status: not deployed by this package. No live app URL is claimed.
Deployment portal: https://share.streamlit.io/

1. Create a GitHub repository and upload this folder's **contents**, keeping
   artifacts/, examples/, .streamlit/, predictor.py and preprocessing.py.
   Do not upload raw banking files, personal information, credentials or secrets.
2. Sign in to https://share.streamlit.io/ with your account and authorise access
   to the repository. Choose **Create app** and deployment from an existing repo.
3. Select your repository and branch. Set the entrypoint to `streamlit_app.py`.
4. In advanced settings, select **Python 3.12**, then deploy.
5. Wait for successful startup and copy the app's actual `streamlit.app` URL.
   Test the three valid demos and invalid demo at that URL before submitting it.

Keep a repo private and use restricted app access where available if permissions
for public model redistribution are unresolved. No training data is in this
bundle, but model publication still requires appropriate reuse permission.

Official instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Question 7 evidence

7.1: Streamlit web UI loads artifacts/classification_pipeline.joblib.
7.2: Synthetic demo, uploaded/pasted CSV and editable manual feature profile;
the full saved preprocessing pipeline returns a class, probabilities and context.
7.3: Schema, type, range and rolling-count checks; clear errors; class/feature
guidance; no personal records, secrets or identifiers in this package.
7.4: Reproducible local command above, deployment steps, and three valid cases
plus one invalid case tested with Streamlit AppTest in evidence/test_results.json.

Run `python test_app.py` to reproduce the UI/model checks.

## Demonstration

Choose Demonstration cases and click Predict demand for each:

* Valid low activity: expected Low (0-49).
* Valid medium activity: expected Medium (50-199).
* Valid high activity: expected High (200+).
* Invalid negative count: expected `month_01 cannot be negative.`

All examples are synthetic. Predicted probabilities are uncalibrated model
estimates. The forecast horizon is November 2015-January 2016; this is a historical
academic demonstration, not a validated current-day operational bank service.

For uploads, use examples/valid_features.csv as the exact schema template.
Do not upload personal or confidential records to an unauthorised hosted app.
The app processes inputs in memory without explicitly saving uploaded files.

## Model and sources

The saved Random Forest had 260 trees, depth 12 and minimum leaf size 6. The
pipeline includes its training-fitted imputation, signed log transform and scaler.
Files are trusted assignment artifacts. Never replace the joblib with an
untrusted file, because deserialisation can execute code.

The original dataset challenge:
https://zindi.africa/competitions/nedbank-transaction-volume-forecasting-challenge

The README states CC BY-SA 4.0; the competition also includes retention terms.
Confirm academic and publication permissions. This app includes no original data.
The module preprocessing.py is retained to load the saved transformation.

AI assistance: initial conversion and tests prepared with ChatGPT/Codex.
Students should review, understand and disclose assistance according to policy.
