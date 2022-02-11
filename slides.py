import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Experiment Results using Isolation Forest from Scikit-learn") 

select_slide = st.sidebar.selectbox(
    "Which slide would you like to navigate to?",
    ("Training Process", "Results")
)

if select_slide == "Training Process":
    st.subheader("Training Process for Isolation Forest") 

    step = st.radio(
     "Choose a step:",
     ('Step 1', 'Step 2', 'Step 3'))

    if step == 'Step 1':
        st.subheader("Step 1:")
        st.write("Sample audio frames using the same parameters \
            as the Autoencoder-based Method.")
        st.write("Each row of the training dataset corresponds to \
            one timestamp (frame), and it contains the features \
            of 5 consecutive frames.")

    elif step == 'Step 2':
        st.subheader("Step 2:")
        st.write("Train the models using the parameter grid below:")
        model_parameter_grid_df = pd.read_csv('parameter grid.csv')
        st.table(model_parameter_grid_df)

    elif step == 'Step 3':
        st.subheader("Step 3:")
        st.write("Use the \"model.decision_function(data)\" function \
            to compute an anomoly score for each row. The smaller the \
            score is, the more likely this frame is an outlier.")
        st.write("Take the negation of the minimum score among all \
            frames in a file as its anomoly score.")

    elif step == 'Step 4':
        st.subheader("Step 4:")
        st.write("Fit the anomoly scores of all the training files \
            using Gamma Distribution. ")
        st.write("Split the distribution with the ratio of 999 to 1 \
            and take the critical value as the threshold. \
            If a file's anomoly score is larger than the threshold, \
            then it is considered containing abnormal sounds.")
    
elif select_slide == "Results":
    st.subheader("Best Results for Isolation Forest") 
    result_df = pd.read_csv('Results.csv')
    col_ref = {'Baseline 1': 'background-color: #ffec8c', 
            'Autoencoder': 'background-color: #ffec8c', 
            'Baseline 2':'background-color: #c2f5ff',
            'MobileNetV2':'background-color: #c2f5ff'}
    st.table(result_df.style.apply(lambda x: pd.DataFrame(col_ref, \
        index=result_df.index, columns=result_df.columns).fillna(''), axis=None))
    
    st.write("Best Parameters for Isolation Forest:")
    st.write("n_estimators: 200")
    st.write("max_samples: 500")
    st.write("max_features: 10")
    st.write("contamination: 0.001")

