# SpamShield — AI Spam Detection System

A production-ready spam detection system with a premium SaaS dashboard, built with Flask, scikit-learn, and Chart.js.

---

## 📁 Project Structure

```
spam-email-project/
├── backend/
│   ├── app.py             # Flask API (all routes)
│   ├── train_model.py     # ML training script
│   ├── preprocess.py      # NLP preprocessing pipeline
│   ├── model.pkl          # (auto-generated on first run)
│   ├── vectorizer.pkl     # (auto-generated on first run)
│   ├── metrics.json       # (auto-generated on first run)
│   └── stats.json         # (auto-generated on first run)
│
├── templates/
│   ├── index.html         # Home page
│   ├── detect.html        # Detection page
│   ├── preprocess.html    # Preprocessing visualizer
│   ├── models.html        # Model comparison
│   └── dashboard.html     # Analytics dashboard
│
├── static/
│   ├── style.css          # Full premium SaaS CSS
│   ├── app.js             # Shared JS utilities
│   └── charts.js          # Chart.js helpers
│
├── dataset/
│   └── spam.csv           # Place Kaggle dataset here
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
- Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- Download `spam.csv` and place it in `dataset/spam.csv`

### 3. Train the model (optional — auto-trains on first run)
```bash
cd backend
python train_model.py
```

### 4. Run the Flask server
```bash
cd backend
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## 🌐 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Home page |
| GET | `/detect` | Detection page |
| GET | `/preprocess-page` | Preprocessing page |
| GET | `/models-page` | Model comparison page |
| GET | `/dashboard-page` | Analytics dashboard |
| POST | `/predict` | Spam/Ham prediction |
| POST | `/preprocess` | NLP pipeline steps |
| GET | `/model-metrics` | All model metrics |
| GET | `/stats` | Dataset statistics |

### POST /predict
```json
Request:  { "message": "Congratulations! You won a free ticket" }
Response: { "prediction": "Spam", "confidence": 95.6, "model": "Best Classifier" }
```

### POST /preprocess
```json
Request:  { "message": "Congratulations! You won a FREE ticket" }
Response: {
  "original_text": "Congratulations! You won a FREE ticket",
  "lowercased": "congratulations! you won a free ticket",
  "tokens": ["congratulations", "you", "won", "a", "free", "ticket"],
  "cleaned_text": ["congratulations", "won", "free", "ticket"],
  "stemmed_text": ["congratul", "won", "free", "ticket"],
  "final_text": "congratul won free ticket"
}
```

---

## 🧠 ML Pipeline

1. **Preprocessing**: lowercase → strip special chars → tokenize → remove stopwords → Porter stem
2. **Vectorization**: TF-IDF (max_features=3000)
3. **Models trained**:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (LinearSVC)
4. **Best model** saved automatically as `model.pkl`

---

## 📊 Dashboard Features

- **Home**: Hero section, feature cards, pipeline overview
- **Detection**: Real-time spam/ham prediction with confidence scores
- **Preprocessing**: Step-by-step NLP pipeline visualization + token table
- **Models**: Accuracy comparison charts, confusion matrices, model cards
- **Dashboard**: KPI cards, pie charts, word cloud, model performance, Power BI integration

---

## 📈 Power BI Integration

On the Dashboard page, enter your Power BI embedded report URL in the input field to load it as an iframe.

---

## 🎨 Design

- **Theme**: Premium SaaS dark-sidebar layout
- **Colors**: Primary #2563eb, Spam #ef4444, Ham #10b981
- **Font**: Inter
- **Charts**: Chart.js 4.4
- **UI**: Bootstrap 5 + custom CSS
