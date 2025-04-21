import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data_fake=pd.read_csv('Fake.csv')
data_true=pd.read_csv('True.csv')

print(data_fake.head())
print(data_true.head())

data_fake["class"]=0
data_true['class']=1

print(data_fake.shape, data_true.shape)

data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i],axis = 0, inplace = True)

    
data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i],axis = 0, inplace = True)
    
print(data_fake.shape, data_true.shape)

data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1

print(data_fake_manual_testing.head(10))
print(data_true_manual_testing.head(10))

data_merge=pd.concat([data_fake, data_true], axis = 0)
print(data_merge.head(10))

print(data_merge.columns)

data=data_merge.drop(['title','subject','date'], axis = 1)


print(data.isnull().sum()) 

data = data.sample(frac = 1)

print(data.head())

data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)

print(data.columns)

print(data.head())

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
print(LR.fit(xv_train, y_train))

pred_lr = LR.predict(xv_test)

print(LR.score(xv_test, y_test))

print (classification_report(y_test, pred_lr))

#Decision Tree classifier

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
print(DT.fit(xv_train, y_train))

pred_dt = DT.predict(xv_test)

print(DT.score(xv_test, y_test))

print (classification_report(y_test, pred_dt))

#Gradient Boost Classifier 

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
print(GB.fit(xv_train, y_train))

pred_gb = GB.predict(xv_test)

print(GB.score(xv_test, y_test))

print(classification_report(y_test, pred_gb))

#random forest classifier

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
print(RF.fit(xv_train, y_train))

pred_rf = RF.predict(xv_test)

print(RF.score(xv_test, y_test))

print (classification_report(y_test, pred_rf))

##Testing model

def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print("\n\nLR Predicition: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(output_label(pred_LR[0]),
                                                                                                             output_label(pred_DT[0]),
                                                                                                             output_label(pred_GB[0]),
                                                                                                             output_label(pred_RF[0])))

#manual testing 

news = str(input("which news you wnt to test: ")) 
manual_testing(news)

import joblib

# Save models
joblib.dump(LR, "logistic_model.pkl")
joblib.dump(DT, "decision_tree_model.pkl")
joblib.dump(GB, "gradient_boost_model.pkl")
joblib.dump(RF, "random_forest_model.pkl")

# Save vectorizer
joblib.dump(vectorization, "tfidf_vectorizer.pkl")

