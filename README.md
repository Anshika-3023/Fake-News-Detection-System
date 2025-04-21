# 📰 Fake News Detection using Machine Learning

This project is a machine learning-based Fake News Detection system that classifies news articles as **Fake News** or **Not A Fake News** using natural language processing and multiple ML models.

---

## 📁 Project Files

| File Name                  | Description                                         |
|---------------------------|-----------------------------------------------------|
| `fake_news_detection.py`  | Trains the model, evaluates it, and saves it using `joblib` |
| `load_and_predict.py`     | Loads the saved models and allows manual testing via user input |
| `Fake.csv`                | Dataset of fake news articles                       |
| `True.csv`                | Dataset of real news articles                       |

---

## 🧠 ML Models Used

- Logistic Regression
- Decision Tree
- Gradient Boosting
- Random Forest

All models are trained using TF-IDF vectorized text data.

---

## 🛠 Requirements

Install required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib

