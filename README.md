# 📱 Sentiment Analysis for Apple iPhone Customer Reviews

## 👤 Author
**El Mahdi El Alj (GH1033521)**  
**Course:** Big Data Analytics (M508)  
**Assignment:** Final Individual Assignment  

---

## 🧩 Problem Statement

Apple iPhones are globally recognized, but customer satisfaction is key to remaining competitive. Analyzing thousands of reviews manually is infeasible—so this project uses **NLP and Machine Learning** to classify reviews into:

- ✅ Positive
- ➖ Neutral
- ❌ Negative

This analysis can help Apple:
- Identify pain points from negative reviews
- Understand customer satisfaction
- Improve future product releases

---

## 🎯 Objectives

- Use traditional ML models and a transformer model (DistilRoBERTa) to classify customer sentiments
- Preprocess, vectorize, balance, and analyze the review data
- Compare model performance with metrics and confusion matrices

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk` for text preprocessing
- `scikit-learn` for ML models and metrics
- `imblearn` for oversampling via SMOTE
- `transformers` and `datasets` from HuggingFace
- `torch` for model training

---

## 🗃️ Dataset

- Source: [Kaggle - iPhone Reviews](https://www.kaggle.com/datasets/mrmars1010/iphone-customer-reviews-nlp/data)
- File used: `iphone.csv`
- Fields: `reviewDescription`, `ratingScore`, etc.

---

## 🧪 Workflow Summary

### 1. **Data Loading**
- Read the dataset using `pandas`

### 2. **Sentiment Mapping**
- Convert `ratingScore` into sentiment labels:
  - 4–5 → Positive
  - 3   → Neutral
  - 1–2 → Negative

### 3. **Preprocessing**
- Lowercasing, punctuation removal
- Tokenization, stopword removal, lemmatization

### 4. **Text Vectorization**
- TF-IDF used to convert text into numerical feature matrix (500 max features)

### 5. **Balancing**
- Applied SMOTE to oversample minority classes

### 6. **Train/Test Split**
- Used 80/20 split

### 7. **Modeling**
- **Traditional Models**: Logistic Regression, Random Forest
- **Advanced Model**: DistilRoBERTa transformer fine-tuned on the data

---

## 📊 Evaluation Metrics

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | ~0.91     | ~0.90      | ~0.91   | ~0.90     |
| DistilRoBERTa       | 0.87     | 0.81 (weighted) | 0.87   | 0.83     |

> Note: DistilRoBERTa underperformed in detecting neutral reviews.

---

## 📉 Confusion Matrix - DistilRoBERTa

           precision    recall  f1-score   support
negative 0.73 0.90 0.80 138
neutral 0.00 0.00 0.00 44
positive 0.92 0.95 0.93 414
accuracy 0.87 596


---

## 🧼 Preprocessing Function

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

📌 Conclusion
This project demonstrates a full NLP pipeline for sentiment analysis using both traditional ML and transformer-based models. Despite the strong performance of DistilRoBERTa, simple models like Logistic Regression still provide excellent accuracy with minimal complexity.
