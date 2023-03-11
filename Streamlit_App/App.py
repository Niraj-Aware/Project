import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Set page title and favicon
st.set_page_config(page_title="My Streamlit App", page_icon=":guardsman:")

# Page title
st.title("Similarity Test App!")

# Add input fields for questions
question1 = st.text_input("Enter your first question:")
question2 = st.text_input("Enter your second question:")

# Add a submit button
if st.button("Submit"):
    # Compare the two questions using the TF-IDF approach
    tfidf_vectorizer = TfidfVectorizer()
    question1_tfidf = tfidf_vectorizer.fit_transform([question1])
    question2_tfidf = tfidf_vectorizer.transform([question2])
    tfidf_similarity = cosine_similarity(question1_tfidf, question2_tfidf)[0][0] * 100

    # Compare the two questions using the BoW approach
    bow_vectorizer = CountVectorizer()
    question1_bow = bow_vectorizer.fit_transform([question1])
    question2_bow = bow_vectorizer.transform([question2])
    bow_similarity = cosine_similarity(question1_bow, question2_bow)[0][0] * 100

    # Compare the two questions using the FuzzyWuzzy approach
    fuzzy_similarity = fuzz.token_sort_ratio(question1, question2)

    # Show the accuracy scores in percentages
    st.write("Similarity scores:")
    st.write(f"TF-IDF: {tfidf_similarity:.2f}%")
    st.write(f"BoW: {bow_similarity:.2f}%")
    st.write(f"FuzzyWuzzy: {fuzzy_similarity:.2f}%")
