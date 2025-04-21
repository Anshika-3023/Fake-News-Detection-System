# 📰 Fake News Detection System using Machine Learning

This project is designed to detect whether a news article is **Fake** or **Real** using multiple machine learning models. It also includes a user-friendly web interface built with **Flask** for manual testing and interaction.

---

## 🚀 Key Features

- Implements models like Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Utilizes TF-IDF for effective feature extraction from textual data.
- Provides an intuitive Flask-based web interface for real-time news validation.
- Clean frontend design using HTML and CSS.
- Includes labeled datasets for both fake and real news articles.

---

## 📁 Folder Structure

```
📦 Fake-News-Detection-System
├── datasets/
│   ├── Fake.csv                # Dataset with fake news
│   └── True.csv                # Dataset with real news
├── static/
│   └── style.css               # Custom CSS for the UI
├── templates/
│   └── index.html              # HTML frontend
├── app.py                      # Main Flask app
├── fake_news_detection.py      # Model training script
├── load_&_predict_model.py     # Script for loading models and predicting
├── *.pkl                       # Trained model and vectorizer files
└── README.md
```

---

## 🛠 How to Run the Project

1. 📥 Clone the repository:
```bash
git clone https://github.com/Anshika-3023/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

2. 📦 Install required Python packages:
```bash
pip install -r requirements.txt
```

3. ▶️ Launch the Flask application:
```bash
python app.py
```

4. 🌐 Open the following URL in your browser:
```
http://127.0.0.1:5000/
```

---

## 📊 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

All models are trained using TF-IDF features extracted from the dataset to classify news articles effectively.

---

## 📚 Dataset

- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains verified real news articles.  
Source: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 👩‍💻 About the Developer

**Anshika Rathore**  
🔗 [LinkedIn](https://www.linkedin.com/in/anshika-rathore-263358263)  
💻 [GitHub](https://github.com/Anshika-3023)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
