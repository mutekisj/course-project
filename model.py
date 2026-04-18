import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def train_model():
    df = pd.read_csv("data/dataset.csv", encoding="utf-8")
    df = df.dropna()

    df["text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2,
        random_state=42
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }

    results = {}
    best_model = None
    best_name = ""
    best_acc = 0

    for name, clf in models.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", clf)
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        results[name] = acc

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    return best_model, best_name, results

def predict(model, text):
    text = preprocess(text)
    return model.predict([text])[0]