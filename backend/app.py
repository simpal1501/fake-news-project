from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Clean text
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


# Prediction function
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

        prediction = predict_news(
            news
        )

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)