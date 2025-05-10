import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
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

param_dist = {
    'n_estimators': [100, 150, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                   n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42, verbose=2)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

print("\nBest parameters found by RandomizedSearchCV:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=['Legitimate', 'Phishing'])

tp = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tn = cm[1, 1]

accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) != 0 else 0
recall_manual = tp / (tp + fn) if (tp + fn) != 0 else 0

print("\nĐánh giá mô hình:")
print(f"- Độ chính xác (Accuracy): {accuracy_manual:.2f}")
print(f"- Độ chính xác dự đoán Phishing (Precision): {precision_manual:.2f}")
print(f"- Khả năng nhận diện Phishing đúng (Recall): {recall_manual:.2f}")

model_path = '../models/random_forest_URL.pkl'
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Model đã được lưu tại: {model_path}")