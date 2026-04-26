import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. LOAD & CLEAN
df = pd.read_csv("data.csv")
df = df.fillna(df.mean(numeric_only=True))

# 2. SEPARATE & SPLIT
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. MODEL: RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. MEASURE
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

