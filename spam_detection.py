import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/spam_data.csv")

# Download stopwords & lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Check if dataset is too small
if len(df) < 5:
    print("❌ ERROR: Dataset is too small! Add more spam and non-spam emails.")
    exit()

# Improved Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply preprocessing
df["Email Text"] = df["Email Text"].astype(str).apply(preprocess_text)

# Convert labels (Spam → 1, Not Spam → 0)
df["Label"] = df["Label"].map({"Spam": 1, "Not Spam": 0})

# Check class balance
print("\nClass Distribution:\n", df["Label"].value_counts())

# Feature Extraction with Improved TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')  
X = vectorizer.fit_transform(df["Email Text"]).toarray()
y = df["Label"]

# Train-Test Split (NO stratify for small dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Tuned Naïve Bayes Model
nb_model = MultinomialNB(alpha=0.1)  # Lower alpha improves performance
nb_model.fit(X_train, y_train)

# Alternative Model: Random Forest Classifier (Higher Accuracy)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
nb_pred = nb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluate Naïve Bayes Model
print("\n📌 Naïve Bayes Model Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# Evaluate Random Forest Model
print("\n📌 Random Forest Model Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Save the Best Model (Random Forest is usually better)
best_model = rf_model if accuracy_score(y_test, rf_pred) > accuracy_score(y_test, nb_pred) else nb_model
pickle.dump(best_model, open("model/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
