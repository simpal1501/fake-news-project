import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = (
    df["title"].fillna("") + " " + df["text"].fillna("")
).apply(clean_text)

X = df["content"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

pickle.dump(model, open("backend/model.pkl", "wb"))
pickle.dump(vectorizer, open("backend/vectorizer.pkl", "wb"))

print("Model saved successfully")