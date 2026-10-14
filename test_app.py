"""Question 7 demonstrations exercise real Streamlit widgets and pipeline."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from predictor import ROOT,predict,load_model

def main():
    tests=[]
    for case,expected in [('Valid low activity','Low:'),('Valid medium activity','Medium:'),('Valid high activity','High:')]:
        at=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
        assert not at.exception,at.exception
        at.selectbox(key='demo_case').select(case).run()
        at.button(key='predict_demo').click().run()
        assert not at.exception and not at.error
        assert expected in at.metric[0].value
        tests.append({'case':case,'passed':True,'predicted_class':at.metric[0].value,'probability':at.metric[1].value})
    at.selectbox(key='demo_case').select('Invalid negative count').run()
    at.button(key='predict_demo').click().run()
    assert not at.exception and 'month_01 cannot be negative' in at.error[0].value
    tests.append({'case':'Invalid negative count','passed':True,'error':at.error[0].value})
    # Exercise the same CSV parser path used by upload through the paste alternative.
    valid=pd.read_csv(ROOT/'examples/valid_features.csv')
    at.radio(key='input_mode').set_value('Upload CSV').run()
    at.text_area(key='pasted_csv').set_value(valid.to_csv(index=False)).run()
    at.button(key='predict_upload').click().run()
    assert not at.exception and not at.error and at.success
    tests.append({'case':'Three-row CSV through Streamlit input','passed':True})
    missing=valid.drop(columns=['month_01'])
    at.text_area(key='pasted_csv').set_value(missing.to_csv(index=False)).run()
    at.button(key='predict_upload').click().run()
    assert not at.exception and 'Missing required features' in at.error[0].value
    tests.append({'case':'Missing required feature','passed':True,'error':at.error[0].value})
    at.radio(key='input_mode').set_value('Manual entry').run()
    at.button(key='predict_manual').click().run()
    assert not at.exception and not at.error and 'Low:' in at.metric[0].value
    tests.append({'case':'Editable manual profile','passed':True,'predicted_class':at.metric[0].value})
    bad=valid.copy();bad.loc[0,'count_3m']=999
    try:predict(bad);raise AssertionError('Inconsistent rolling count accepted')
    except ValueError as e:assert 'must equal the sum' in str(e)
    tests.append({'case':'Inconsistent rolling count','passed':True})
    personal=valid.copy();personal['CustomerName']='Example'
    try:predict(personal);raise AssertionError('Extra personal field accepted')
    except ValueError as e:assert 'extra columns' in str(e)
    tests.append({'case':'Extra identifier column blocked','passed':True})
    # Confirm no reimplementation bypasses preprocessing.
    actual=predict(valid);prob=load_model().predict_proba(valid)
    assert np.allclose(actual[['low_probability','medium_probability','high_probability']],prob)
    tests.append({'case':'Complete saved pipeline probabilities preserved','passed':True})
    output={'model_sha256':hashlib.sha256((ROOT/'artifacts/classification_pipeline.joblib').read_bytes()).hexdigest(),'checks':tests}
    (ROOT/'evidence/test_results.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
