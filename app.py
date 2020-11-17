# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 02:02:23 2020

@author: chra8017
"""

# Core Pkgs
import streamlit as st
import os

# ML Pkgs

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import base64

# EDA Pkgs
import numpy as np
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 

# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Data Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns

def main ():
    Activities = ['Run SVC Model','Data Exploration']
    
    title_temp ="""
    	<div style="background-color:#407294;padding:10px;border-radius:10px;margin:10px;font-weight:bold;">
    	<h1 style="color:white;text-align:center;">Linear Support Vector Classifier</h1>
    	</div>
    	"""
    st.markdown(title_temp,unsafe_allow_html=True)
        
    def file_uploader(key):
        uploaded_file  = st.file_uploader("",key=key)
        return uploaded_file

    choice = st.sidebar.selectbox('Select Activity',Activities)
    if choice == 'Run SVC Model':
#================================UPLOAD TRAINING DATA=========================#  
    
        title_temp ="""
    	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
    	<h3 style="color:white;text-align:center;">Upload The Training Data File</h3>
    	</div>
    	"""
        
        st.markdown(title_temp, unsafe_allow_html=True)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        upload_file_training = file_uploader(0)
        
        if upload_file_training is not None:    
            df = pd.read_excel(upload_file_training)
            df = df.astype(str)
            df_y = df.drop(columns='ItemDescription')
            
            if st.sidebar.checkbox("View Training Data"):
                st.dataframe(df_y.head(6))
    
#==========================UPLOADING THE TEST DATA============================#
    
        title_temp ="""
    	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
    	<h3 style="color:white;text-align:center;">Upload The Test Data File</h3>
    	</div>
    	"""
        
        st.markdown(title_temp, unsafe_allow_html=True)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        upload_file_test = file_uploader(1)
        if upload_file_test is not None:
            st.set_option('deprecation.showfileUploaderEncoding', False)
            test_data = pd.read_excel(upload_file_test)
            test_data = test_data.astype(str)
            test_input = test_data['#AU LOC: ITEM_DESCRIPTION_AU']
        
        
            if st.sidebar.checkbox("View Test Data"):
                st.dataframe(test_input.head(6))
        
            all_columns = df.columns.tolist()
    
#========================USER DEFINED FEATURE VARIABLES=======================#
            
            df_list_X = st.multiselect("Decide X Variable",all_columns)
            print(df_list_X)
            df_list_y = st.multiselect("Decide Y Variable",all_columns)
            print(df_list_y)
    
#=========================MACHINE LEARNING ALGORITHM==========================#
#        @st.cache(suppress_st_warning=True,allow_output_mutation=True)
#        def run_algorithm ():
            if len(df_list_X) != 0 and len(df_list_y) !=0:
                X = df[df_list_X[0]]
                y = df_y[df_list_y[0]]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
                pipeline = Pipeline([('vect', CountVectorizer()),('svc', CalibratedClassifierCV(base_estimator=LinearSVC()))])
                model = pipeline.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                df_class = pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)).transpose()
                df_class.reset_index(level=0, inplace=True)
                df_class.rename(columns = {'index':'CATEGORY'}, inplace = True) 
                title_temp ="""<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;"><h3 style="color:white;text-align:center;">Classification Report</h3></div>"""
                st.markdown(title_temp, unsafe_allow_html=True)
                st.dataframe(df_class)    
                y_pred = model.predict(test_input)
                probs = model.predict_proba(test_input)
                class_indexes = np.argmax(probs,axis=1)
                max_probs = probs[np.arange(len(test_input)),class_indexes]
                    
                test_data[('prediction'+'_'+df_y['TP_Segment'].name)] = pd.Series(y_pred, index=test_data.index)
                test_data[('max_probs'+'_'+df_y['TP_Segment'].name)] = pd.Series(max_probs, index=test_data.index)
                
                csv = test_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}" download="TPNew Items 0518 0601.csv">Download The Final Outcome</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    if choice == 'Data Exploration':
        title_temp ="""
    	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
    	<h3 style="color:white;text-align:center;">Upload the Input Data</h3>
    	</div>
    	"""
        
        st.markdown(title_temp, unsafe_allow_html=True)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        upload_file_training = file_uploader(0)
        
        if upload_file_training is not None:    
            df = pd.read_excel(upload_file_training)
            df = df.astype(str)
            #df_y = df.drop(columns='ItemDescription')
            
            if st.sidebar.checkbox("Data Highlights"):
                st.dataframe(df.head(6))
        
            title_temp ="""<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
                            <h3 style="color:white;text-align:center;">Data Analysis</h3></div>"""
            st.markdown(title_temp, unsafe_allow_html=True)
            profile = ProfileReport(df)
            st_profile_report(profile)
    
#Initate the the APP

if __name__ == '__main__':
    main()     
        
#    if st.button("Classify"):
#    X = df['ItemDescription']
#    y = df_y['TP_Segment']
#
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
#    pipeline = Pipeline([('vect', CountVectorizer()),
#                   ('svc', CalibratedClassifierCV(base_estimator=LinearSVC()))])
#
#    model = pipeline.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    df_class = pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)).transpose()
#    df_class.reset_index(level=0, inplace=True)
#    df_class.rename(columns = {'index':'CATEGORY'}, inplace = True) 
#    st.dataframe(df_class)
#    
##    df_model_train = (model.score(X_train, y_train))
##    st.dataframe(df_model_train)
##    
##    df_model_test = (model.score(X_test, y_test))
##    st.dataframe(df_model_train)
#    
#    y_pred = model.predict(test_input)
#    probs = model.predict_proba(test_input)
#    class_indexes = np.argmax(probs,axis=1)
#    max_probs = probs[np.arange(len(test_input)),class_indexes]
#        
#    test_data[('prediction'+'_'+df_y['TP_Segment'].name)] = pd.Series(y_pred, index=test_data.index)
#    test_data[('max_probs'+'_'+df_y['TP_Segment'].name)] = pd.Series(max_probs, index=test_data.index)
#    
#    csv = test_data.to_csv(index=False)
#    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#    href = f'<a href="data:file/csv;base64,{b64}" download="TPNew Items 0518 0601.csv">Download The Final Outcome</a>'
#    st.markdown(href, unsafe_allow_html=True)

#st.markdown("<h1 style='text-align: center; color: blue;'>Linear Support Vector Classifier</h1>", unsafe_allow_html=True)



#================================UPLOAD TRAINING DATA=========================#

#title_temp ="""
#	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
#	<h3 style="color:white;text-align:center;">Upload The Training Data File</h3>
#	</div>
#	"""
#st.markdown(title_temp, unsafe_allow_html=True)
#
#st.set_option('deprecation.showfileUploaderEncoding', False)
#
#uploaded_file = st.file_uploader("",key=0)
#
#if uploaded_file is not None:
##    path = 'C:/Users/chra8017/Desktop/AVI ML WEB APP'
##    os.chdir(path)
#    df = pd.read_excel(uploaded_file, sheet_name='Training Data')
#    df = df.astype(str)
#    df_y = df.drop(columns='ItemDescription')
#
#
##    title_temp ="""
##	<div style="background-color:#153169;padding:10px;border-radius:10px;margin:10px;">
##	<h5 style="color:white;text-align:center;">Training Data Highlights</h5>
##	</div>
##	"""
#    if st.checkbox("View Training Data"):
#        st.dataframe(df_y.head(6))

##==========================UPLOADING THE TEST DATA============================#
#
#title_temp ="""
#	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
#	<h3 style="color:white;text-align:center;">Upload The Test Data File</h3>
#	</div>
#	"""
#st.markdown(title_temp, unsafe_allow_html=True)
#
#st.set_option('deprecation.showfileUploaderEncoding', False)
#
#uploaded_file = st.file_uploader("",key=1)
#
#
#if uploaded_file is not None:
#    test_data = pd.read_excel(uploaded_file)
#    test_data = test_data.astype(str)
#    test_input = test_data['#AU LOC: ITEM_DESCRIPTION_AU']
##    title_temp ="""
##	<div style="background-color:#153169;padding:10px;border-radius:10px;margin:10px;">
##	<h5 style="color:white;text-align:center;">Test Data Highlights</h5>
##	</div>"""
#    if st.checkbox('View Test Data'):
##        st.markdown(title_temp, unsafe_allow_html=True)
#        st.dataframe(test_input.head(6))
#
#if st.checkbox("Select Columns for Classifier"):
#    df_selections = df.columns()
#    selected_columns = st.multiselect('Select Feature Variables',df_selections)
#    new_df = df['selected_columns']
#    st.dataframe(new_df)
#
#if st.button("Classify"):
#    st.spinner("The Machine is Learning....")
#    X = df['ItemDescription']
#    y = df_y['TP_Segment']
#
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
#    pipeline = Pipeline([('vect', CountVectorizer()),
#                   ('svc', CalibratedClassifierCV(base_estimator=LinearSVC()))])
#
#    model = pipeline.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    df_class = pd.DataFrame(classification_report(y_test, y_pred,output_dict=True)).transpose()
#    df_class.reset_index(level=0, inplace=True)
#    df_class.rename(columns = {'index':'CATEGORY'}, inplace = True) 
#    st.dataframe(df_class)
#    
##    df_model_train = (model.score(X_train, y_train))
##    st.dataframe(df_model_train)
##    
##    df_model_test = (model.score(X_test, y_test))
##    st.dataframe(df_model_train)
#    
#    y_pred = model.predict(test_input)
#    probs = model.predict_proba(test_input)
#    class_indexes = np.argmax(probs,axis=1)
#    max_probs = probs[np.arange(len(test_input)),class_indexes]
#        
#    test_data[('prediction'+'_'+df_y['TP_Segment'].name)] = pd.Series(y_pred, index=test_data.index)
#    test_data[('max_probs'+'_'+df_y['TP_Segment'].name)] = pd.Series(max_probs, index=test_data.index)
#    
#    csv = test_data.to_csv(index=False)
#    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#    href = f'<a href="data:file/csv;base64,{b64}" download="TPNew Items 0518 0601.csv">Download The Final Outcome</a>'
#    st.markdown(href, unsafe_allow_html=True)
#    
#    title_temp ="""
#	<div style="background-color:#40948c;padding:10px;border-radius:10px;margin:10px;">
#	<h3 style="color:white;text-align:center;">Classification Report Plot (Precision Vs Support)</h3>
#	</div>
#	"""
#    data = df_class.head(10)
#    st.markdown(title_temp, unsafe_allow_html=True)
#    st.write(sns.barplot(x='precision', y='support', data=data, hue = 'CATEGORY'))
#    st.pyplot()
