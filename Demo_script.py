import re
import string
import joblib
from nltk.corpus import stopwords


lr = joblib.load("optimized_logistic_regression_model.pkl")
cv1 = joblib.load("tfidf_vectorizer.pkl")


alphanumeric = lambda x: re.sub(r'\w*\d\w*|\d+', ' ', x)
remove_html = lambda x: re.sub(r'<.*?>', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
stop = stopwords.words('english')

def preprocess_text(text):
    text = alphanumeric(text)
    text = remove_html(text)
    text = punc_lower(text)
    text = remove_n(text)
    text = remove_non_ascii(text)
    text = ' '.join([word for word in text.split() if word not in stop])
    return text


review = input("Enter a movie review :")

cleaned_review = preprocess_text(review)
review_vector = cv1.transform([cleaned_review])
prediction = lr.predict(review_vector)[0]

label = "Bad Review" if prediction == 0 else "Good Review"
print(f"\nPrediction: {label}")