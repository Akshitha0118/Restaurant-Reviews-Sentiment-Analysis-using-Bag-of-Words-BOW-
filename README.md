# Restaurant-Reviews-Sentiment-Analysis-using-Bag-of-Words-BOW

🍽️ Restaurant Reviews Sentiment Analysis using Bag of Words (BOW)
📌 Project Overview

This project performs Sentiment Analysis on restaurant reviews using Natural Language Processing (NLP) techniques and multiple Machine Learning classification models.
The goal is to compare different algorithms based on training accuracy, testing accuracy, bias, and variance.

🧠 Key Concepts Used

Text Preprocessing (Regex, Tokenization, Stemming)

Stopword Removal (NLTK)

Feature Extraction using Bag of Words (CountVectorizer)

Supervised Machine Learning Models

Bias–Variance Analysis

📂 Dataset

Restaurant_Reviews.tsv

1000 customer reviews

Binary sentiment labels:

1 → Positive

0 → Negative

🛠️ Text Preprocessing Steps

Removed non-alphabetic characters

Converted text to lowercase

Tokenized words

Removed English stopwords

Applied Porter Stemmer

Reconstructed cleaned text corpus

🔢 Feature Engineering

Bag of Words (BOW) using CountVectorizer

Converted text into numerical feature vectors

🤖 Machine Learning Models Implemented

The following classifiers were trained and evaluated:

Decision Tree

Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest

Support Vector Machine (Linear SVM)

XGBoost

Naive Bayes

LightGBM

📊 Model Evaluation Metrics

For each model, the following were calculated:

Training Accuracy

Testing Accuracy

Bias = 1 − Training Accuracy

Variance = Training Accuracy − Testing Accuracy

This helps in understanding overfitting and underfitting behavior of each algorithm.

📈 Results

Each model’s performance is compared based on accuracy, bias, and variance to identify the most reliable classifier for sentiment prediction.

🚀 Technologies & Libraries

Python

Pandas, NumPy

Matplotlib

NLTK

Scikit-learn

XGBoost

LightGBM

▶️ How to Run the Project
pip install -r requirements.txt
python app.py


(Or run directly in Jupyter Notebook)

🎯 Conclusion

This project demonstrates how different ML models behave on text data using BOW features and highlights the importance of bias-variance tradeoff in model selection.
