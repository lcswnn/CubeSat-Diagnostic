import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# PUT DATA WITHIN DATA FOLDER TO READ -- CHANGE THIS AS NEEDED TO TRAIN MODEL WITH DIFFERENT DATASETS.
df = pd.read_csv("../data/dataset.csv")

# --- Derived features (computed from existing per-window stats) ---
def add_derived_features(frame):
    f = frame.copy()

    # Coefficient of variation: std relative to mean magnitude
    f['cv'] = f['std'] / (f['mean'].abs() + 1e-8)

    # Ratio of diff variance to overall variance (captures temporal instability)
    f['diff_var_ratio'] = f['diff_var'] / (f['var'] + 1e-8)
    f['diff2_var_ratio'] = f['diff2_var'] / (f['var'] + 1e-8)

    # Peak density: peaks per unit length
    f['peak_density'] = f['n_peaks'] / (f['len'] + 1e-8)
    f['smooth10_peak_density'] = f['smooth10_n_peaks'] / (f['len'] + 1e-8)
    f['smooth20_peak_density'] = f['smooth20_n_peaks'] / (f['len'] + 1e-8)

    # Smoothing peak ratio: how much structure survives smoothing
    f['smooth10_ratio'] = f['smooth10_n_peaks'] / (f['n_peaks'] + 1e-8)
    f['smooth20_ratio'] = f['smooth20_n_peaks'] / (f['n_peaks'] + 1e-8)

    # Kurtosis × skew interaction (jointly captures tail behavior)
    f['kurtosis_skew'] = f['kurtosis'] * f['skew']

    # Mean-to-variance ratio
    f['mean_var_ratio'] = f['mean'] / (f['var'] + 1e-8)

    return f

df = add_derived_features(df)

train_df = df[df['train'] == 1]
test_df = df[df['train'] == 0]

drop_cols = ['segment', 'anomaly', 'train', 'channel']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['anomaly']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['anomaly']

model = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(model, 'model.pkl')