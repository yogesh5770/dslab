import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cybersecurity_attacks.csv")

print("Data Loaded",df.head())
print("missing values before the preprocessing",df.isnull().sum())
df = df.fillna(df.mean(numeric_only = True))

cat_cols = df.select_dtypes(include = ['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = le.fit_transform(df[col])


sns.violinplot(data=df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

df.hist()
plt.tight_layout()
plt.show()
