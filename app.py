from flask import Flask, render_template, request
import joblib
import re
import string

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

@app.route("/", methods=["GET", "POST"])
def index():
    final_prediction = None
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)
        vectorized = vectorizer.transform([cleaned])

        # Raw model predictions (0 = Fake, 1 = Real)
        predictions = [
            LR.predict(vectorized)[0],
            DT.predict(vectorized)[0],
            GB.predict(vectorized)[0],
            RF.predict(vectorized)[0]
        ]

        if any(pred == 1 for pred in predictions):
            final_prediction = "Real News"
        else:
            final_prediction = "Fake News"

    return render_template("index.html", prediction=final_prediction)

if __name__ == "__main__":
    app.run(debug=True)

