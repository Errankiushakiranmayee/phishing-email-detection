import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("="*50)
    print("🚀 Starting Phishing Email Detection Model Pipeline")
    print("="*50)

    # 1. Define Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'email_dataset.csv')
    matrix_output_path = os.path.join(base_dir, 'confusion_matrix.png')

    # 2. Load the Dataset
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)
    
    text_col = 'body'
    label_col = 'label'

    # Drop rows where the label itself is missing
    df = df.dropna(subset=[label_col])

    # Clean text content column cleanly
    df[text_col] = df[text_col].fillna("").astype(str)
    
    print(f"✅ Data loaded successfully. Total Samples: {len(df)}")
    print(f"📊 Class distribution:\n{df[label_col].value_counts()}\n")

    # 3. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    # 4. Feature Extraction (Extracts textual keywords, suspicious phrases, and embedded URL signs)
    print("🧪 Extracting textual features & embedded URL structures via TF-IDF...")
    
    # Using a word-level vectorizer that automatically parses text links and phishing keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Train the Scikit-Learn Model
    print("🤖 Training the Machine Learning Classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 6. Model Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🎯 EXPERIMENT RESULTS: Model Accuracy = {accuracy:.2%}")
    print("="*50)
    print("\n📝 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Generate and Save Confusion Matrix Plot
    print("📊 Generating Visual Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(l) for l in np.unique(y_test)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Phishing Email Detection - Confusion Matrix', fontsize=14, pad=15, weight='bold')
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.tight_layout()
    
    plt.savefig(matrix_output_path, dpi=300)
    plt.close()
    print(f"💾 Confusion Matrix graph successfully saved as: {matrix_output_path}\n")
    print("🎉 Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()