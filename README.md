# Phishing Email Detection Model

A Machine Learning project built with Python and Scikit-Learn that preprocesses textual email logs and classifies them as either "Phishing" or "Safe" using natural language processing techniques.

## 🚀 Key Features

* **Real-world Dataset Evaluation:** Trained on a standardized email dataset parsing thousands of samples.
* **Natural Language Processing (NLP):** Implements `TfidfVectorizer` to extract, clean, and weigh key textual phrases, patterns, and embedded URL signatures.
* **Efficient Classification:** Powered by a optimized Logistic Regression model for highly accurate, production-speed classification.
* **Performance Mapping:** Generates a visually mapped Confusion Matrix and precision-recall reports.

* 📁 **phishing-email-detection/**
    * 📁 **data/**
        * 📄 `email_dataset.csv` — Sourced raw email dataset
    * 📁 **src/**
        * 🐍 `main.py` — Core preprocessing and model pipeline
    * 📊 `confusion_matrix.png` — Evaluated model performance chart
    * ℹ️ `README.md` — Project documentation and setup guide
    * 📄 `requirements.txt` — Python environment dependencies