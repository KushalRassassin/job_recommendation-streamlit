# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For creating web applications
import numpy as np  # For numerical operations
import pickle  # For serializing and deserializing Python objects
import string  # For string manipulation

# Read data from a CSV file into a pandas DataFrame, skipping bad lines and using the Python engine
df1 = pd.read_csv('new.csv', on_bad_lines='skip', engine='python')

# Drop rows with missing values in the DataFrame
df1 = df1.dropna()

# Import TfidfVectorizer from scikit-learn for text vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer with English stop words
tdif = TfidfVectorizer(stop_words='english')

# Fill missing values in the 'jobdescription' column with an empty string
df1['jobdescription'] = df1['jobdescription'].fillna('')

# Apply TfidfVectorizer to the 'jobdescription' column to get the TF-IDF matrix
tdif_matrix = tdif.fit_transform(df1['jobdescription'])
tdif_matrix.shape

# Import sigmoid_kernel from scikit-learn for calculating similarity scores
from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the cosine similarity matrix using sigmoid_kernel
cosine_sim = sigmoid_kernel(tdif_matrix, tdif_matrix)

# Create a Series with job title indices for later reference
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

# Import the operator module for sorting purposes
import operator

# Define a function to get job recommendations based on similarity scores
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=operator.itemgetter(1), reverse=True)  # Sort based on similarity values
    sim_scores = sim_scores[4:16]  # Select the top 15 most similar job titles
    tech_indices = [int(i[0]) for i in sim_scores]
    recommended_jobs = df1.iloc[tech_indices]['jobtitle'].tolist()
    return recommended_jobs

# Create a Streamlit web application header
st.header('tech jobs recommender')

# Load pre-trained job data and similarity scores
jobs = pickle.load(open('job_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# Extract job titles from the loaded data
jobs_list = jobs['jobtitle'].values

# Create a dropdown menu for selecting a job title
selected_job = st.selectbox(
    "Type or select a job from the dropdown",
    jobs_list
)

# Display a button to trigger the recommendation process
if st.button('Show Recommendation'):
    # Get job recommendations and display them as subheaders
    recommend_job_names = get_recommendations(selected_job)
    for i in recommend_job_names:
        st.subheader(i)
