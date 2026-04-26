
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the data
df = pd.read_csv("cybersecurity_attacks.csv")

# 2. Select columns for NLP
# Using 'Payload Data' to predict 'Attack Type'
X = df['Payload Data'].astype(str) 
y = df['Attack Type']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text Vectorization (TF-IDF)
# We limit features to 5000 to keep the model efficient and remove stop words (common words like 'the', 'is')
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Model Training (Naive Bayes)
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 6. Prediction and Evaluation
y_pred = nb_model.predict(X_test_tfidf)

print("--- Text Classification Report ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

msg = input("Enter log or message to analyze: ")
msg_transformed = tfidf.transform([msg])
prediction = nb_model.predict(msg_transformed)[0]

print(f"🚨 Security Alert Type: {prediction.upper()}")