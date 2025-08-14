#  Movie Review Sentiment Classifier

##  Overview
This project builds a **sentiment analysis model** that predicts whether a movie review is **positive** or **negative**.  
It takes raw text reviews, cleans and processes them using Natural Language Processing (NLP) techniques, and then applies multiple machine learning models to find the best-performing one.

---

##  Approach
1. **Data Loading**  
   - Downloaded the training and testing datasets using `gdown`.
   - Loaded CSV files into Pandas DataFrames.

2. **Data Exploration**  
   - Checked dataset size, structure, and missing values.
   - Visualized sentiment distribution using pie charts (Matplotlib & Seaborn).

3. **Text Preprocessing**  
   - Removed HTML tags, punctuation, numbers, and non-ASCII characters.
   - Converted all text to lowercase.
   - Removed stopwords with NLTK.
   - Applied tokenization and stemming (Lancaster Stemmer).

4. **Feature Extraction**  
   - Used **TF-IDF Vectorization** (unigrams, English stopwords removed) to convert text into numerical features.

5. **Model Training**  
   Trained and compared the following classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Bernoulli Naive Bayes
   - Multinomial Naive Bayes
   - Linear Support Vector Classifier (LinearSVC)
   - Random Forest

6. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.
   - Compared all models to identify the top performer.

---

##  Tools & Libraries
- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **NLTK** (stopwords, stemming)
- **Scikit-learn** (ML models & evaluation metrics)
- **gdown** (for dataset download)

---

##  Results
- **Best Model**: `LinearSVC` — achieved the highest accuracy and F1-score.
- **Key Insight**: Using TF-IDF features significantly improved results compared to raw counts.
- Logistic Regression and Random Forest also performed well, but slightly below LinearSVC.

---

## How to Run
1. Open this file in **Jupyter Notebook**.
2. From the menu, select **Cell → Run All**.
3. The notebook will:
   - Download the dataset.
   - Preprocess the text.
   - Train multiple models.
   - Show performance results and visualizations.

---



