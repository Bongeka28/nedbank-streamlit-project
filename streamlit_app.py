"""Streamlit Question 7 interface using the saved optimised classifier."""
import io
import numpy as np
import pandas as pd
import streamlit as st
from predictor import ROOT,COLS,LABELS,load_model,predict

st.set_page_config(page_title='Banking demand forecast',page_icon='📊',layout='wide')
st.title('Banking demand forecast')
st.caption('IDA117V · Historical customer-demand classification · Saved optimised Random Forest')
st.warning('Academic demonstration using 2015 data. Model probabilities are uncalibrated estimates. Use for capacity-planning research only; do not use for lending, fraud allegations, service denial or eligibility decisions.')

@st.cache_resource
def model():return load_model()

examples=pd.read_csv(ROOT/'examples/valid_features.csv')
with st.sidebar:
    st.header('Demand classes')
    st.write('**Low:** 0-49 future transactions')
    st.write('**Medium:** 50-199 future transactions')
    st.write('**High:** 200 or more future transactions')
    st.caption('Forecast horizon: November 2015-January 2016. Thresholds are project assumptions, not official bank workload categories.')
    st.info('Use synthetic examples on a public demo. Do not submit names, IDs, account numbers or confidential banking records.')
    st.caption('Files are processed in server memory and are not written to disk by this app. A hosted service still receives uploaded bytes.')

mode=st.radio('Input method',['Demonstration cases','Upload CSV','Manual entry'],horizontal=True,key='input_mode')
frame=None;clicked=False
if mode=='Demonstration cases':
    st.subheader('Try a synthetic case')
    st.write('These examples contain invented values and no personal records.')
    case=st.selectbox('Case',['Valid low activity','Valid medium activity','Valid high activity','Invalid negative count'],key='demo_case')
    if case=='Invalid negative count':frame=pd.read_csv(ROOT/'examples/invalid_features.csv')
    else:frame=examples.iloc[['Valid low activity','Valid medium activity','Valid high activity'].index(case):][:1].copy()
    with st.expander('Inspect the input features'):
        st.dataframe(frame.T.rename(columns={frame.index[0]:'Value'}),width='stretch')
    clicked=st.button('Predict demand',type='primary',key='predict_demo')
elif mode=='Upload CSV':
    st.subheader('Upload prepared customer features')
    st.write('Use the supplied 55-feature schema. This accepts engineered customer features, not raw bank statements. Limit: 1,000 rows and 2 MB.')
    st.download_button('Download valid CSV template',examples.to_csv(index=False),'valid_features.csv','text/csv')
    st.download_button('Download invalid demonstration', (ROOT/'examples/invalid_features.csv').read_bytes(),'invalid_features.csv','text/csv')
    uploaded=st.file_uploader('Feature CSV',type=['csv'],key='feature_csv')
    text=st.text_area('Or paste CSV text',help='Use either the file upload or pasted CSV, not both.',key='pasted_csv')
    clicked=st.button('Predict uploaded rows',type='primary',key='predict_upload')
    if clicked:
        try:
            if uploaded is not None and text.strip():raise ValueError('Choose a file upload or pasted CSV, not both.')
            if uploaded is not None:
                if uploaded.size>2*1024*1024:raise ValueError('File exceeds 2 MB.')
                frame=pd.read_csv(uploaded)
            elif text.strip():
                if len(text.encode())>2*1024*1024:raise ValueError('Pasted CSV exceeds 2 MB.')
                frame=pd.read_csv(io.StringIO(text))
            else:raise ValueError('Upload a CSV or paste CSV text first.')
        except (ValueError,pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError):
            st.error('Could not read the input. Choose one valid UTF-8 CSV under 2 MB with the required feature columns.');clicked=False
else:
    st.subheader('Enter one customer feature profile')
    st.write('Start with a synthetic template and edit the values. Keep rolling counts consistent with the monthly counts. Blank optional financial/income values use the saved training imputer.')
    profile=st.selectbox('Starting template',['Low activity','Medium activity','High activity'],key='manual_profile')
    ix=['Low activity','Medium activity','High activity'].index(profile)
    initial=pd.DataFrame({'Feature':COLS,'Value':examples.iloc[ix][COLS].to_numpy(dtype=float)})
    edited=st.data_editor(initial,disabled=['Feature'],hide_index=True,num_rows='fixed',width='stretch',key='manual_editor_'+str(ix))
    frame=pd.DataFrame([dict(zip(edited.Feature,edited.Value))])
    clicked=st.button('Predict entered profile',type='primary',key='predict_manual')

if clicked and frame is not None:
    try:
        result=predict(frame,model())
        st.success('Prediction completed')
        if len(result)==1:
            row=result.iloc[0]
            c1,c2=st.columns(2);c1.metric('Predicted class',row.predicted_class);c2.metric('Model probability',f'{row.probability:.1%}')
            st.write(row.explanation)
            st.dataframe(pd.DataFrame({'Class':['Low','Medium','High'],'Probability':[row.low_probability,row.medium_probability,row.high_probability]}),hide_index=True,width='stretch')
        else:
            st.dataframe(result,hide_index=True,width='stretch')
        st.download_button('Download predictions',result.to_csv(index=False),'predictions.csv','text/csv')
    except ValueError as e:st.error(str(e))

with st.expander('Feature guidance and limitations'):
    st.write('month_01 through month_12 represent November 2014 through October 2015. Counts are nonnegative whole numbers. Amount and balance fields use ZAR. debit_share is between 0 and 1; flags use 0 or 1.')
    st.dataframe(pd.read_csv(ROOT/'feature_guidance.csv'),hide_index=True,width='stretch')
    st.write('The saved pipeline performs training-fitted median imputation, signed log transformation, scaling and classification. No models are trained during prediction. Historical accuracy does not establish reliability on current customers or measured service benefits. Missing months indicate zero observed records, not necessarily inactivity.')
    st.caption('Model provenance: the unchanged assignment champion, selected on validation data. Confirm dataset/model reuse terms before publishing. The app bundle includes no raw records or real customer examples.')
