import streamlit as st
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect
from collections import Counter
import spacy
import numpy as np

# Function to scrape comments from YouTube using the YouTube Data API
def scrape_comments(video_url, api_key):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    video_id = video_url.split("=")[-1]
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )

        while request:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            request = youtube.commentThreads().list_next(request, response)
    except Exception as e:
        st.error("Error fetching comments:", e)
        return None

    return comments

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Function to perform sentiment analysis using NLTK
def nltk_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    preprocessed_text = preprocess_text(text)
    sentiment = sia.polarity_scores(preprocessed_text)
    return sentiment['compound']

# Function to detect language of the text
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang

# Function to extract named entities using spaCy
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return doc.ents

# Function to provide suggestions based on NER analysis
def provide_suggestions(entities):
    suggestions = []

    persons = [entity.text for entity in entities if entity.label_ == "PERSON"]
    if persons:
        suggestions.append(f"Your viewers mentioned {', '.join(persons)} in the comments. Consider engaging with them directly by responding to their comments.")

    organizations = [entity.text for entity in entities if entity.label_ == "ORG"]
    if organizations:
        suggestions.append(f"Your video seems to have attracted the attention of {', '.join(organizations)}. Consider collaborating with them for future videos.")

    locations = [entity.text for entity in entities if entity.label_ == "GPE"]
    if locations:
        suggestions.append(f"Your viewers are mentioning locations such as {', '.join(locations)}. Consider creating content related to these locations to further engage your audience.")

    return suggestions

# Function to summarize text using TextBlob
def summarize_text(text):
    blob = TextBlob(text)
    summary = blob.sentences[:2]  # Get the first two sentences as summary
    return ' '.join(str(sentence) for sentence in summary)

# Streamlit UI
st.title("YouTube Comment Analysis")

video_url = st.text_input("Enter YouTube Video URL:")
api_key = st.text_input("Enter YouTube Data API Key:")

if st.button("Scrape Comments"):
    if not video_url or not api_key:
        st.warning("Please provide both the YouTube Video URL and API key.")
    else:
        comments = scrape_comments(video_url, api_key)

        if comments is None:
            st.error("Failed to fetch comments for the video. Please check the video URL and API key.")
        elif not comments:
            st.warning("No comments found for the video.")
        else:
            st.success("Comments scraped successfully!")
            st.write("Here are the first few comments:")
            st.write(comments[:5])

            # Perform sentiment analysis
            sentiment_scores = [nltk_sentiment(comment) for comment in comments]
            overall_sentiment_score = np.mean(sentiment_scores)
            video_rating = round(overall_sentiment_score * 5, 2)
            st.write("\nGeneral Review of the Video:")
            if overall_sentiment_score >= 0.7:
                st.write("The video received overwhelmingly positive feedback.")
            elif 0.5 <= overall_sentiment_score < 0.7:
                st.write("The video received mostly positive feedback.")
            elif 0.3 <= overall_sentiment_score < 0.5:
                st.write("The video received mixed feedback.")
            elif 0.1 <= overall_sentiment_score < 0.3:
                st.write("The video received mostly negative feedback.")
            else:
                st.write("The video received overwhelmingly negative feedback.")
            st.write("\nVideo Rating (out of 5):", video_rating)

            # Plot overall sentiment distribution
            st.write("\nOverall Sentiment Distribution:")
            plt.hist(sentiment_scores, bins=20, color='skyblue')
            plt.title('Overall Sentiment Distribution')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            st.pyplot(plt)

            # Extract entities from comments and provide suggestions
            all_entities = [entity for comment in comments for entity in extract_entities(comment)]
            st.write("\nSuggestions based on Named Entity Recognition (NER):")
            if not all_entities:
                st.write("No entities detected in the comments.")
            else:
                suggestions = provide_suggestions(all_entities)
                if suggestions:
                    st.write("\n".join(suggestions))

            # Plot sentiment vs count
            sentiment_counts = pd.Series(sentiment_scores).apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')).value_counts().sort_index()
            st.write("\nSentiment vs Count:")
            plt.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
            plt.title('Sentiment vs Count')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            st.pyplot(plt)

            # Segregate comments into positive, negative, and neutral categories
            positive_comments = [comment for comment, sentiment in zip(comments, sentiment_scores) if sentiment >= 0.05]
            negative_comments = [comment for comment, sentiment in zip(comments, sentiment_scores) if sentiment <= -0.05]
            neutral_comments = [comment for comment, sentiment in zip(comments, sentiment_scores) if -0.05 < sentiment < 0.05]

            # Summarize comments
            st.write("\nSummarized Positive Comments:")
            for i, comment in enumerate(positive_comments[:5]):
                st.write(f"Comment {i+1}: {summarize_text(comment)}")

            st.write("\nSummarized Negative Comments:")
            for i, comment in enumerate(negative_comments[:5]):
                st.write(f"Comment {i+1}: {summarize_text(comment)}")

            st.write("\nSummarized Neutral Comments:")
            for i, comment in enumerate(neutral_comments[:5]):
                st.write(f"Comment {i+1}: {summarize_text(comment)}")
