# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IZg7qgkP0YDEQinpFpqPocL1bFnZ_KGO
"""

import pandas as pd
import streamlit as st
import numpy as np
import pickle
import string

df1 = pd.read_csv('new.csv', on_bad_lines='skip', engine='python')
df1 = df1.dropna()

from sklearn.feature_extraction.text import TfidfVectorizer

tdif = TfidfVectorizer(stop_words='english')

df1['jobdescription'] = df1['jobdescription'].fillna('')

tdif_matrix = tdif.fit_transform(df1['jobdescription'])
tdif_matrix.shape

from sklearn.metrics.pairwise import sigmoid_kernel
cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_sim):
  idx = indices[title]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda X:X[1], reverse=True)
  sim_scores = sim_scores[1:16]
  tech_indices = [i[0] for i in sim_scores]
  return df1['jobtitle'].iloc[tech_indices]

st.header('tech jobs recommender')
jobs = pickle.load(open('job_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

jobs_list = jobs['jobtitle'].values
selected_job = st.selectbox(
    "Type or select a job from the dropdown",
    jobs_list
)
if st.button('Show Recommendtion'):
    recommend_job_names = get_recommendations(selected_job)
    for i in recommend_job_names:
        st.subheader(i)
