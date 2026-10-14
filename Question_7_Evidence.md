# Question 7 - Streamlit implementation and evidence

## 7.1 Working interface
The Streamlit app loads the unchanged optimised Random Forest and its complete saved preprocessing pipeline. It is a real web application, launched with `python -m streamlit run streamlit_app.py`.

## 7.2 Input, prediction and explanation
Users can upload a feature CSV, paste CSV text, edit a single feature profile, or choose a synthetic demo. Results show the class, model probability, all class probabilities and a short activity-context explanation. No causal attribution is claimed.

## 7.3 Validation and responsible use
Input checks cover schema, nonnumeric values, negative/fractional counts, infinite values, range-invalid flags and inconsistent rolling totals. Extra columns are rejected, including identifiers. Feature guidance and historical-use limitations appear in the interface. The app includes no raw data, real customer records or credentials. It uses synthetic demo inputs and processes requests in memory.

## 7.4 Launch, deployment and demonstrations
Local commands:
```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```
Local URL: http://localhost:8501 (only your computer, not a public link).
Public status: not deployed. The deployment portal presented a sign-in page, and GitHub was not connected. README.md provides the exact repository/entrypoint/Python settings needed for Streamlit Community Cloud.
Deployment portal: https://share.streamlit.io/ . It is not the app demonstration URL.

| Demonstration | Actual outcome | Passed |
| --- | --- | --- |
| Valid low activity | Low: 0-49 transactions (81.2%) | Yes |
| Valid medium activity | Medium: 50-199 transactions (85.6%) | Yes |
| Valid high activity | High: 200+ transactions (80.7%) | Yes |
| Invalid negative count | month_01 cannot be negative. | Yes |
| Three-row CSV through Streamlit input | Check succeeded | Yes |
| Missing required feature | Missing required features: month_01 | Yes |
| Editable manual profile | Low: 0-49 transactions | Yes |
| Inconsistent rolling count | Check succeeded | Yes |
| Extra identifier column blocked | Check succeeded | Yes |
| Complete saved pipeline probabilities preserved | Check succeeded | Yes |

The first four checks exercise actual Streamlit UI widgets through AppTest. The three-row CSV test exercises the same parser via the paste input; it does not claim a browser file-upload test. The saved pipeline probability check ensures inference applies the full original preprocessing.

Before submission, replace the pending deployment status with the real URL only after successful publishing and rerun demonstrations at that URL.

Official sources: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app and https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest .

AI-assisted implementation; review and disclose under institutional policy.