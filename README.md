## 📰 Fake News Classifier using Machine Learning
A machine learning project to classify whether a news article is Fake or Real using **Natural Language Processing (NLP)** and models like Logistic Regression, Multinomial Naive Bayes, and Random Forest Classifier. Built with explainability and interview-readiness in mind.

---

## 📁 Dataset
We used the publicly available Fake and Real News Dataset, consisting of:

- Fake.csv – 23,481 fake news articles
- True.csv – 21,417 true news articles

After preprocessing and merging:

- ✅ Total rows: 44,898
- ✅ Final data used for modeling (after cleaning): 8,980 samples (balanced)

---

## 🔧 Project Workflow

### Loading and Merging:
- Combined Fake.csv and True.csv, labeled them (1 for fake, 0 for true).
### Text Cleaning & Preprocessing:
- Removed punctuation, special characters, and stopwords
- Lowercased text
- Applied stemming for normalization
- Used TF-IDF Vectorizer to convert text into numerical features
- Model Training: We trained and evaluated the following models:
 - Logistic Regression
 - Multinomial Naive Bayes
 - Random Forest Classifier
### Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve
- Feature Importance (for Random Forest)
### Model Selection & Saving: 
- Random Forest was the best performer and was saved using joblib for future use.

---

## 📊 Model Comparison

|         Model	         | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
|  Logistic Regression	 |  98.81%	|   0.99	|  0.99  |   0.99   |
|Multinomial Naive Bayes |	94.53%	|   0.96	|  0.94  |   0.95   |
|Random Forest Classifier|	99.83%	|   1.00	|  1.00  |   1.00   |  
### 🔥 Best Performing Model: Random Forest

---

## Best Parameters (via GridSearchCV):

```bash
{
  'n_estimators': 200,
  'max_depth': None,
  'min_samples_split': 2
}
```

---

## Confusion Matrix:

```bash
True Positives  = 4222
True Negatives  = 4758
False Positives = 0
False Negatives = 0
ROC-AUC Score: ~1.00
```

---

## Feature Importance: 
- Top TF-IDF words contributing most to classification were plotted.

---

## 💾 Model Saving & Loading
```bash
import joblib

# Save model
joblib.dump(rf_model, 'best_model_random_forest.pkl')

# Load model
# loaded_model = joblib.load('best_model_random_forest.pkl')
# predictions = loaded_model.predict(X_test)
```

---

## 🌐 Flask Deployment
- Built a web interface using Flask to interact with the trained model.
- Steps:
 - Built a simple form where users can paste news text.
 - On submission, the text is preprocessed and passed to the loaded Random Forest model.
 - The result (Fake or Real) is returned on the same page with styling.
 - Run the Flask App:
   ```bash
   python app.py
   ```
   Then open http://127.0.0.1:5000/ in your browser.

## 📌 Tools & Libraries Used
- Python
 - Pandas, NumPy
 - Scikit-learn
 - Matplotlib, Seaborn
 - joblib, wordcloud
 - NLTK
- Flask
- Jupyter Notebook