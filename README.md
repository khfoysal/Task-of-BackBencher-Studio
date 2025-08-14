# Movie Review Sentiment Classifier

## 📌 Overview
This project implements a **movie review sentiment classifier** that predicts whether a review expresses a **positive** or **negative** sentiment.  
It processes raw text data, applies NLP preprocessing, and evaluates multiple machine learning algorithms to identify the best-performing model.

---

## 🛠 Approach
1. **Data Loading**  
   - Downloaded training and testing datasets using `gdown`.
   - Loaded CSV data with Pandas.

2. **Data Exploration**  
   - Checked dataset info, size, and missing values.
   - Visualized sentiment distribution with Matplotlib and Seaborn.

3. **Text Preprocessing**  
   - Removed HTML tags, punctuation, numbers, and non-ASCII characters.
   - Converted text to lowercase.
   - Removed stopwords using NLTK.
   - Applied tokenization and stemming (Lancaster Stemmer).

4. **Feature Extraction**  
   - Used `TfidfVectorizer` (unigrams, English stopwords removal).

5. **Model Training**  
   Trained and evaluated multiple classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Bernoulli Naive Bayes
   - Multinomial Naive Bayes
   - Linear Support Vector Classifier (LinearSVC)
   - Random Forest

6. **Evaluation Metrics**  
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC-AUC score

---

## 🧰 Tools & Libraries
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **NLTK** (stopwords, stemming)
- **Scikit-learn** (ML models & evaluation metrics)
- **gdown** (dataset download)

---

## 📊 Results
- **Best Performing Model**: `LinearSVC` (highest accuracy & F1-score)
- TF-IDF vectorization significantly improved results compared to raw counts.
- Models like Logistic Regression and Random Forest also performed well but slightly below LinearSVC.

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook Movie_Review_Classifier.ipynb
