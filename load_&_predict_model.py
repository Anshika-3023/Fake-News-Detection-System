# load_and_predict.py

import joblib
import pandas as pd
import string
import re

# Load saved models and vectorizer
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boost_model.pkl")
RF = joblib.load("random_forest_model.pkl")
vectorization = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Label output
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Manual testing
def manual_testing(news):
    sample = {"text": [news]}
    new_data = pd.DataFrame(sample)
    new_data['text'] = new_data['text'].apply(wordopt)
    new_x = new_data['text']
    new_xv = vectorization.transform(new_x)

    print("\n--- Prediction Results ---")
    print(f"Logistic Regression: {output_label(LR.predict(new_xv)[0])}")
    print(f"Decision Tree      : {output_label(DT.predict(new_xv)[0])}")
    print(f"Gradient Boosting  : {output_label(GB.predict(new_xv)[0])}")
    print(f"Random Forest      : {output_label(RF.predict(new_xv)[0])}")

# Run manual prediction
news = input("Enter news text to test: ")
manual_testing(news)
