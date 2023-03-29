import streamlit as st
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Define a function to calculate the accuracy score
def get_accuracy_score(approach, input_str1, input_str2):
    X = [input_str1 + ' ' + input_str2] # Combine the two input strings into one document
    y = ['similar'] # This is just a placeholder label since we only have one document
    if approach == 'BoW':
        vectorizer = CountVectorizer()
    elif approach == 'TF-IDF':
        vectorizer = TfidfVectorizer()
    else:
        return fuzz.ratio(input_str1, input_str2) / 100 # Divide by 100 to get a score between 0 and 1
    X_transformed = vectorizer.fit_transform(X)
    clf = RandomForestClassifier()
    clf.fit(X_transformed, y)
    y_pred = clf.predict(X_transformed)
    accuracy_score = clf.score(X_transformed, y)
    return accuracy_score

# Define the Streamlit app
def app():
    st.title('Similar Questions Pair')
    st.write('Enter two questions to determine if they are semantically equivalent:')
    input_str1 = st.text_input('Question 1')
    input_str2 = st.text_input('Question 2')
    if st.button('Submit'):
        st.write('Bag-of-Words (BoW) accuracy score:', get_accuracy_score('BoW', input_str1, input_str2))
        st.write('Term Frequency-Inverse Document Frequency (TF-IDF) accuracy score:', get_accuracy_score('TF-IDF', input_str1, input_str2))
        st.write('Fuzzy Wuzzy accuracy score:', get_accuracy_score('Fuzzy', input_str1, input_str2))

if __name__ == '__main__':
    app()
