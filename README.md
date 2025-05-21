# Sentiment Analysis for Apple iPhone Customer Reviews

## Author
**El Mahdi El Alj (GH1033521)**  
**Course:** Big Data Analytics M508  
**Assignment:** Final Individual Assignment  

---

## 📌 Business Problem

The iPhone is among the most popular smartphones globally. Due to significant market competition, understanding customer feedback is essential. This project applies sentiment analysis to classify customer reviews into **positive**, **neutral**, or **negative** sentiments. This classification enables Apple to:

- Improve customer satisfaction
- Identify areas of improvement
- Recognize appreciated features

---

## 💡 Objective

Use machine learning (NLP pipeline) to:
- Automate classification of iPhone reviews
- Extract insights on user satisfaction
- Guide business and product decisions

---

## 🧰 Tools and Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `nltk` (for text preprocessing)
- `sklearn` (for ML models and preprocessing)
- `datasets` (for data handling)
- `imblearn` (for oversampling with SMOTE)
- `transformers` (if used later in the notebook)

---

## 📥 Dataset

The dataset `iphone.csv` contains Apple iPhone customer reviews from Amazon India.

---

## ⚙️ Workflow Overview

1. **Load and Explore Dataset**
2. **Select Relevant Columns**  
   Focus on `reviewDescription` and `ratingScore`
3. **Handle Missing Values**
4. **Sentiment Mapping**  
   - Ratings ≥
