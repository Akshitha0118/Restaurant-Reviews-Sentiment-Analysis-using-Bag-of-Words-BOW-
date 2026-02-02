# BOW original dataset /without stop words


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# Text Preprocessing (WITH STOPWORDS)
# -----------------------------
import re
import nltk
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0, 1000):
    review=re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

    
from sklearn.feature_extraction.text import CountVectorizer

tfidf = CountVectorizer()
X = tfidf.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values



# -----------------------------
# Train-Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# =================================================
# 1️⃣ Decision Tree
# =================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)

y_pred_dt = classifier_dt.predict(X_test)

train_dt = classifier_dt.score(X_train, y_train)
test_dt = accuracy_score(y_test, y_pred_dt)

bias_dt = 1 - train_dt
variance_dt = train_dt - test_dt

train_dt, test_dt, bias_dt, variance_dt


# =================================================
# 2️⃣ Logistic Regression
# =================================================
from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(max_iter=1000)
classifier_lr.fit(X_train, y_train)

y_pred_lr = classifier_lr.predict(X_test)

train_lr = classifier_lr.score(X_train, y_train)
test_lr = accuracy_score(y_test, y_pred_lr)

bias_lr = 1 - train_lr
variance_lr = train_lr - test_lr

train_lr, test_lr, bias_lr, variance_lr


# =================================================
# 3️⃣ KNN
# =================================================
from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=5)
classifier_knn.fit(X_train, y_train)

y_pred_knn = classifier_knn.predict(X_test)

train_knn = classifier_knn.score(X_train, y_train)
test_knn = accuracy_score(y_test, y_pred_knn)

bias_knn = 1 - train_knn
variance_knn = train_knn - test_knn

train_knn, test_knn, bias_knn, variance_knn


# =================================================
# 4️⃣ Random Forest
# =================================================
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_rf.fit(X_train, y_train)

y_pred_rf = classifier_rf.predict(X_test)

train_rf = classifier_rf.score(X_train, y_train)
test_rf = accuracy_score(y_test, y_pred_rf)

bias_rf = 1 - train_rf
variance_rf = train_rf - test_rf

train_rf, test_rf, bias_rf, variance_rf


# =================================================
# 5️⃣ SVM (Linear)
# =================================================
from sklearn.svm import LinearSVC

classifier_svm = LinearSVC()
classifier_svm.fit(X_train, y_train)

y_pred_svm = classifier_svm.predict(X_test)

train_svm = classifier_svm.score(X_train, y_train)
test_svm = accuracy_score(y_test, y_pred_svm)

bias_svm = 1 - train_svm
variance_svm = train_svm - test_svm

train_svm, test_svm, bias_svm, variance_svm


# =================================================
# 6️⃣ XGBoost
# =================================================
from xgboost import XGBClassifier

classifier_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=0
)
classifier_xgb.fit(X_train, y_train)

y_pred_xgb = classifier_xgb.predict(X_test)

train_xgb = classifier_xgb.score(X_train, y_train)
test_xgb = accuracy_score(y_test, y_pred_xgb)

bias_xgb = 1 - train_xgb
variance_xgb = train_xgb - test_xgb

train_xgb, test_xgb, bias_xgb, variance_xgb


# =================================================
# 7️⃣ Naive Bayes
# =================================================
from sklearn.naive_bayes import GaussianNB

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred_nb = classifier_nb.predict(X_test)

train_nb = classifier_nb.score(X_train, y_train)
test_nb = accuracy_score(y_test, y_pred_nb)

bias_nb = 1 - train_nb
variance_nb = train_nb - test_nb

train_nb, test_nb, bias_nb, variance_nb


# =================================================
# 8️⃣ LightGBM
# =================================================
from lightgbm import LGBMClassifier

classifier_lgbm = LGBMClassifier()
classifier_lgbm.fit(X_train, y_train)

y_pred_lgbm = classifier_lgbm.predict(X_test)

train_lgbm = classifier_lgbm.score(X_train, y_train)
test_lgbm = accuracy_score(y_test, y_pred_lgbm)

bias_lgbm = 1 - train_lgbm
variance_lgbm = train_lgbm - test_lgbm

train_lgbm, test_lgbm, bias_lgbm, variance_lgbm
