import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("../data/dataset.csv")

train_df = df[df['train'] == 1]
test_df = df[df['train'] == 0]

drop_cols = ['segment', 'anomaly', 'train', 'channel']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['anomaly']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['anomaly']

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(model, 'model.pkl')