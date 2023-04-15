import streamlit as st
import pandas as pd
from google.cloud import firestore
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud



# your google credentials json file 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "spring-appliedml-ab9851a06569.json"

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("Sentiment Analyzer")

home, model= st.tabs(["Home", "Model"])

with home:
    # getting data from your firestore database - reddit collection

    df = pd.read_csv('modified_test.csv')

    posts_length_range = st.sidebar.slider("Posts Length", min_value=1, max_value=9999, value=[1, 9999])
    
    df['length'] = df['review'].apply(lambda x: len(x))
    df = df.loc[(df.length >= posts_length_range[0]) & (df.length <= posts_length_range[1]), :]
    
    chart, desc = st.columns([2,1])
    with chart: 
        fig = px.histogram(df, x='length', color='rating', color_discrete_map={"1": "blue", "0": "tomato"}, barmode="group")
        st.plotly_chart(fig)
        st.caption("sentiment on patient reviews on UCI Machine Learning Repository")
    with desc:
        st.subheader("Patient Review Sentiment on on UCI Machine Learning Repository")
        st.write("In the bar chart on the left, a rating of 1 indicates \
                 that the patient rating falls within the range of 7 to 10, \
                 representing a positive attitude towards the drugs. \
                 A rating of 0 indicates that the patient rating falls within \
                 the range of 1 to 6, representing a negative attitude towards \
                 the drugs.")

    "---"

    # World Cloud
    df_pos = df.loc[df.rating == 1, ["review"]]
    df_neg = df.loc[df.rating == 0, ["review"]]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Posts")
        st.image(WordCloud().generate("/n".join(list(df_pos.review))).to_image())
    with col2:
        st.subheader("Negative Posts")
        st.image(WordCloud().generate("/n".join(list(df_neg.review))).to_image())

    "---"

    st.subheader("Sample Posts")
    if st.button("Show Sample Posts and Sentiment"):
        placeholder = st.empty()
        with placeholder.container():
            for index, row in df.sample(10).iterrows():
                text = row["review"].strip()
                if text != "":
                    col1, col2 = st.columns([3,1])
                    with col1:
                        with st.expander(text[:100] + "..."):
                            st.write(text)
                    with col2:
                        if row["rating"] == 1:
                            st.info(row['rating'])
                        else:
                            st.error(row['rating'])
        if st.button("Clear", type="primary"):
            placeholder.empty()

with model:
    st.subheader("Our Sentiment Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "82%", "3%")
    col2.metric("F1 Score", "0.78", "0.03")
    with col3:
        "We have been able to achieve this amazing performance in our sentiment models \
        using the bhah bhah features and techniques bhah bhah"
    
    st.subheader("Model Training Dataset")
    train = pd.read_csv('train.csv')
    st.write(train)
    "---"
    st.subheader("Modified Training Dataset")
    modified_train = pd.read_csv('modified_train.csv')
    st.write(modified_train)
    "---"
    st.subheader("Model Training")
    "---"
    st.subheader("Model Testing")
    "---"
    st.subheader("Evaluation Tuning and Performance")
    "---"
    st.subheader("Final Model")
    "---"
    st.subheader("Data Collection and Workflow")
    "---"
    
