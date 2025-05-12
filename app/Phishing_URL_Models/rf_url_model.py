import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

data_path = '../datasetURL/dataURLphishing.csv'
data = pd.read_csv(data_path)

if 'Label' not in data.columns:
    raise ValueError("Dataset không chứa cột 'Label'")

data = data.dropna(subset=['Label'])
data = data[data['Label'].isin([0, 1])]

label_mapping = {0: 'Legitimate', 1: 'Phishing'}
data['Label'] = data['Label'].map(label_mapping)

X = data.drop(columns=['Label', 'URL'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=['Legitimate', 'Phishing'])

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) != 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) != 0 else 0

print(f"- Độ chính xác (Accuracy): {accuracy_manual:.2f}")
print(f"- Độ chính xác dự đoán Phishing (Precision): {precision_manual:.2f}")
print(f"- Khả năng nhận diện Phishing đúng (Recall): {recall_manual:.2f}")

model_path = '../models/random_forest_URL.pkl'
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"Model đã được lưu tại: {model_path}")
