import pandas as pd
import numpy as np
import nltk
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from preprocess import preprocess_text

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def train_and_save():
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'spam.csv')
    
    try:
        df = pd.read_csv(dataset_path, encoding='latin-1')
    except FileNotFoundError:
        print("[INFO] Dataset not found. Generating synthetic data for demo...")
        spam_msgs = [
            "Congratulations! You won a free lottery ticket worth $1000. Call now!",
            "FREE prize waiting for you! Claim your reward immediately!",
            "URGENT: Your account will be suspended. Click here to verify now.",
            "You have been selected for a cash prize of $5000. Reply WIN to claim.",
            "Get rich quick! Make $500 per day from home. Limited offer!",
            "Your mobile number has won a Nokia N90 and £2000 prize. Call 09061701461",
            "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed",
            "Free entry in 2 a wkly comp to win FA Cup final tkts",
            "SMS. ac FREEMSG: Txt: CALL to No: 86888 & claim your reward",
            "Win a £1000 cash prize or a prize worth £5000",
        ] * 25
        ham_msgs = [
            "Hey, are you coming to the meeting tomorrow?",
            "Can you please call me back when you get a chance?",
            "I will be late for dinner tonight, sorry.",
            "The project deadline is next Friday. Please be prepared.",
            "Happy birthday! Hope you have a wonderful day.",
            "Did you see the game last night? It was amazing!",
            "Let me know if you need anything from the store.",
            "Thanks for the help yesterday, really appreciate it.",
            "Can we reschedule our appointment to next week?",
            "Just checking in to see how you are doing.",
        ] * 45
        labels = ['spam'] * len(spam_msgs) + ['ham'] * len(ham_msgs)
        df = pd.DataFrame({'label': labels, 'text': spam_msgs + ham_msgs})

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    elif 'label' not in df.columns:
        df.columns = ['label', 'text'] + list(df.columns[2:])
        df = df[['label', 'text']]

    df = df.dropna(subset=['label', 'text'])
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    df['processed'] = df['text'].apply(preprocess_text)

    X = df['processed']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    metrics = {}
    best_accuracy = 0
    best_model = None
    best_model_name = ''

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm
        }

        print(f"{name}: Accuracy={acc}%, Precision={prec}%, Recall={rec}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print(f"\nBest model: {best_model_name} ({best_accuracy}%)")

    base = os.path.dirname(__file__)
    joblib.dump(best_model, os.path.join(base, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(base, 'vectorizer.pkl'))

    with open(os.path.join(base, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    spam_count = int((df['label'] == 'spam').sum())
    ham_count = int((df['label'] == 'ham').sum())
    stats = {
        'total': len(df),
        'spam': spam_count,
        'ham': ham_count,
        'spam_rate': round(spam_count / len(df) * 100, 2)
    }
    with open(os.path.join(base, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print("Model, vectorizer, metrics, and stats saved successfully.")
    return metrics

if __name__ == '__main__':
    train_and_save()