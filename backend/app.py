from flask import Flask, request, render_template
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")
# Labels
fake["label"] = 0     # Fake
true["label"] = 1     # Real

# Combine and shuffle
df = pd.concat([fake, true], ignore_index=True)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Text cleaning
def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# Combine title + article text
df["content"] = (
    df["title"].fillna("")
    + " "
    + df["text"].fillna("")
)

# Clean content
df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text to numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1,2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(
    X_train_vec,
    y_train
)


def predict_news(news):

    news = clean_text(news)

    news_vec = vectorizer.transform(
        [news]
    )

    pred = model.predict(
        news_vec
    )[0]

    if pred == 1:
        return "Real News ✅"

    return "Fake News ❌"


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        prediction = predict_news(news)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)