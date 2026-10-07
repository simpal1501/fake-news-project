# 📰 Veritas – Fake News Detection & Text Verification System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to detect misinformation and verify the authenticity of news articles in real time.

🔗 **Live Demo:** [https://fake-news-project-s94c.onrender.com/](https://fake-news-project-s94c.onrender.com/)  
💻 **Repository:** [https://github.com/simpal1501/fake-news-project](https://github.com/simpal1501/fake-news-project)

---

## 🚀 Key Features

- **Text Cleaning & Preprocessing:** Automatic URL removal, special character stripping, and lowercase normalization using regular expressions (`re`).
- **TF-IDF Vectorization:** Transforms raw unstructured text into numerical feature representations suitable for ML classification.
- **Machine Learning Classifier:** Employs an optimized Logistic Regression classifier trained on labeled news datasets, achieving $\sim$93% test accuracy.
- **Real-Time Web Interface:** Lightweight Flask-based web application with an intuitive UI for instant text classification.
- **Model Serialization:** Uses `pickle` to serialize and load the trained model and vectorizer for fast inference.

---

## 🛠️ Tech Stack

- **Machine Learning & NLP:** Python, Scikit-learn, TF-IDF Vectorizer, Logistic Regression
- **Web Framework:** Flask, Jinja2 Templates
- **Frontend:** HTML5, CSS3
- **Data & Serialization:** Pandas, NumPy, Pickle
- **Deployment:** Render

---

## 🧠 How It Works (Pipeline)

```
[User Input Text] 
       │
       ▼
[Text Cleaning & Regex Normalization]
       │
       ▼
[TF-IDF Feature Extraction]
       │
       ▼
[Logistic Regression Classification]
       │
       ▼
[Prediction: Real News ✅ / Fake News ❌]
```

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/simpal1501/fake-news-project.git
   cd fake-news-project/backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access in browser:**
   ```
   http://127.0.0.1:5000
   ```

---

## 👤 Author

- **Simpal Kumari**  
  - GitHub: [@simpal1501](https://github.com/simpal1501)  
  - LinkedIn: [Simpal Kumari](https://www.linkedin.com/in/simpal-kumari-76b050310/)
