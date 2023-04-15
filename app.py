import streamlit as st
import pandas as pd
from google.cloud import firestore
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
import re

train = pd.read_csv('modified_train.csv')
train_X = train.loc[:, ['review']]
train_y = train.rating
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, stratify=train_y)
X_train_docs = [doc for doc in X_train.review]
pipeline = Pipeline([
            ('vect', TfidfVectorizer(ngram_range=(1,2), 
                                    stop_words='english')),
            ('cls', LinearSVC())
])
pipeline.fit(X_train_docs, y_train)
training_accuracy = cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()
predicted = pipeline.predict([doc for doc in X_test.review])
validation_accuracy = metrics.accuracy_score(y_test, predicted)
f1_score = f1_score(y_test, predicted)

# your google credentials json file 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "spring-appliedml-ab9851a06569.json"

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("Sentiment Analyzer")

home, model= st.tabs(["Home", "Model"])

with home:
    # getting data from your firestore database - reddit collection
    db = firestore.Client()
    query = db.collection(u'reddit').order_by(u'created', direction=firestore.Query.DESCENDING)
    posts = list(query.stream())
    docs_dict = list(map(lambda x: x.to_dict(), posts))
    df = pd.DataFrame(docs_dict)

    created_end = datetime.fromtimestamp(df.iloc[:1,:].created.values[0])
    created_start = datetime.fromtimestamp(df.iloc[-1:,:].created.values[0])

    date_start = st.sidebar.date_input("From", value=created_start, min_value=created_start, max_value=created_end)
    date_end = st.sidebar.date_input("To", value=created_end, min_value=created_start, max_value=created_end)
    posts_length_range = st.sidebar.slider("Posts Length", min_value=1, max_value=9999, value=[1, 9999])

    date_start_str = date_start.strftime('%Y-%m-%d')
    date_end_str = date_end.strftime('%Y-%m-%d')
    df['date'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    df = df.loc[(df.date >= date_start_str) & (df.date <= date_end_str), :]
    
    df['length'] = df['selftext'].apply(lambda x: len(x))
    df = df.loc[(df.length >= posts_length_range[0]) & (df.length <= posts_length_range[1]), :]
    
    chart, desc = st.columns([2,1])
    with chart: 
        fig = px.histogram(df, x='date', color='sentiment', color_discrete_map={"positive": "blue", "negative": "tomato"}, barmode="group")
        st.plotly_chart(fig)
        st.caption("sentiment on subreddit r/movies")
    with desc:
        st.subheader("Patient Review Sentiment on on UCI Machine Learning Repository")
        st.write("Using our amazing sentiment model we have been able predit \
                 the sentiment of user posts on subreddit 'r/movies'. The graph shows the \
                 number user posts (positive and negative) for each day for the selected date range.")

    "---"

    # World Cloud
    df_pos = df.loc[df.sentiment == "positive", ["selftext"]]
    df_neg = df.loc[df.sentiment == "negative", ["selftext"]]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Posts")
        st.image(WordCloud().generate("/n".join(list(df_pos.selftext))).to_image())
    with col2:
        st.subheader("Negative Posts")
        st.image(WordCloud().generate("/n".join(list(df_neg.selftext))).to_image())

    "---"

    st.subheader("Sample Posts")
    if st.button("Show Sample Posts and Sentiment"):
        placeholder = st.empty()
        with placeholder.container():
            for index, row in df.sample(10).iterrows():
                text = row["selftext"].strip()
                if text != "":
                    col1, col2 = st.columns([3,1])
                    with col1:
                        with st.expander(text[:100] + "..."):
                            st.write(text)
                    with col2:
                        if row["sentiment"] == "positive":
                            st.info(row['sentiment'])
                        else:
                            st.error(row['sentiment'])
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
    
