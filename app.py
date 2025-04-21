from flask import Flask, render_template, request
import joblib
import pandas as pd
import string
import re

app = Flask(__name__)

# Load models and vectorizer
LR = joblib.load("logistic_model.pkl")
DT = joblib.load("decision_tree_model.pkl")
GB = joblib.load("gradient_boost_model.pkl")
RF = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vectorized = vectorizer.transform([cleaned])
        prediction = {
            "lr": output_label(LR.predict(vectorized)[0]),
            "dt": output_label(DT.predict(vectorized)[0]),
            "gb": output_label(GB.predict(vectorized)[0]),
            "rf": output_label(RF.predict(vectorized)[0]),
        }
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
