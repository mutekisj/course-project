import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# -------------------
# nltk стоп-слова
# -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# -------------------
# очистка текста
# -------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------
# загрузка данных
# -------------------
df = pd.read_csv("data/dataset.csv", encoding="utf-8")

df = df.dropna()
df["text"] = df["text"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42
)

# -------------------
# модели
# -------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}

best_model = None
best_name = ""
best_acc = 0

# -------------------
# обучение моделей
# -------------------
for name, clf in models.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", clf)
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results[name] = acc

    print("\n======================")
    print(name)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# -------------------
# график сравнения
# -------------------
plt.bar(results.keys(), results.values())
plt.title("Сравнение моделей")
plt.ylabel("Accuracy")
plt.show()

# -------------------
# confusion matrix (лучшая модель)
# -------------------
pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred_best)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title(f"Confusion Matrix: {best_name}")
plt.show()

# -------------------
# чат-предсказатель
# -------------------
print("\nЛучшая модель:", best_name)

while True:
    text = input("\nДокумент: ")
    text = preprocess(text)
    print("Категория:", best_model.predict([text])[0])