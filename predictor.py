"""Inference and validation for the unchanged assignment champion pipeline."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import preprocessing  # Required by the saved FunctionTransformer.

ROOT=Path(__file__).resolve().parent
COLS=json.loads((ROOT/'artifacts/feature_schema.json').read_text())
LABELS=['Low: 0-49 transactions','Medium: 50-199 transactions','High: 200+ transactions']
MONTHS=[f'month_{i:02d}' for i in range(1,13)]

def load_model():
    return joblib.load(ROOT/'artifacts/classification_pipeline.joblib')

def validate(frame):
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    missing=set(COLS)-set(frame.columns)
    if missing:
        raise ValueError('Missing required features: '+', '.join(sorted(missing)))
    extra=set(frame.columns)-set(COLS)
    if extra:
        raise ValueError('Remove extra columns, including any identifiers or personal data: '+', '.join(sorted(extra)))
    if not 1<=len(frame)<=1000:
        raise ValueError('Provide between 1 and 1,000 feature rows.')
    x=frame.loc[:,COLS].copy()
    counts=[c for c in COLS if c.startswith(('month_','count_','category_')) or c in ['lifetime_count','account_count','active_days','observed_days','recency_days','active_months_12m','fin_products','fin_rows']]
    nonnegative=[c for c in COLS if c.startswith(('month_','count_','mean_','std_','category_')) or c in ['lifetime_count','account_count','active_days','recency_days','observed_days','active_months_12m','fin_products','fin_rows','income_log']]
    for c in COLS:
        num=pd.to_numeric(x[c],errors='coerce')
        if (x[c].notna()&num.isna()).any():
            raise ValueError(f'{c} must contain numbers or empty cells.')
        if np.isinf(num).any():raise ValueError(f'{c} must be finite.')
        if c in nonnegative and (num<0).any():raise ValueError(f'{c} cannot be negative.')
        if c in counts and (num.dropna()%1!=0).any():raise ValueError(f'{c} must contain whole counts.')
        if c in ['income_missing','financials_missing','primary_banking'] and not num.dropna().isin([0,1]).all():
            raise ValueError(f'{c} must be 0 or 1.')
        x[c]=num
    if x[MONTHS].isna().any().any():raise ValueError('All 12 monthly counts are required.')
    if not x.debit_share.dropna().between(0,1).all():raise ValueError('debit_share must lie between 0 and 1.')
    if not x.active_months_12m.dropna().between(0,12).all():raise ValueError('active_months_12m must lie between 0 and 12.')
    for w in [3,6,12]:
        observed=x[MONTHS[-w:]].sum(axis=1)
        supplied=x[f'count_{w}m']
        if supplied.isna().any():raise ValueError(f'count_{w}m is required.')
        if not np.allclose(supplied,observed):raise ValueError(f'count_{w}m must equal the sum of its last {w} monthly counts.')
    return x

def predict(frame,model=None):
    x=validate(frame)
    if model is None:model=load_model()
    probabilities=model.predict_proba(x)
    output=[]
    for i,p in enumerate(probabilities):
        k=int(np.argmax(p));recent=int(x.iloc[i].count_3m)
        output.append({'row':i+1,'predicted_class':LABELS[k],'probability':float(p[k]),
            'low_probability':float(p[0]),'medium_probability':float(p[1]),'high_probability':float(p[2]),
            'explanation':f'The model predicts {LABELS[k].lower()} for November 2015-January 2016. The supplied August-October activity is {recent} transactions. This recent count is context, not a causal explanation. The model uses all validated features.'})
    return pd.DataFrame(output)
