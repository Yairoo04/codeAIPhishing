import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from feature_extraction import extract_features

# Load dataset
data_path = '../dataset_URL/phishing_URL.csv'
data = pd.read_csv(data_path)

if 'URL' not in data.columns or 'Label' not in data.columns:
    raise ValueError("Dataset không chứa các cột bắt buộc: 'URL' và 'Label'")

data = data.dropna(subset=['URL', 'Label']) 
data = data[data['Label'].isin([0, 1])]

# Gán nhãn: 0 - Legitimate, 1 - Phishing
label_mapping = {0: 'Legitimate', 1: 'Phishing'}
data['Label'] = data['Label'].map(label_mapping)

features_list = []
valid_labels = []

for url, label in zip(data['URL'], data['Label']):
    features = extract_features(url)
    if features and all(value != -1 for value in features.values()):
        features_list.append(features)
        valid_labels.append(label)

if not features_list:
    raise ValueError("Không có URL hợp lệ nào được trích xuất. Kiểm tra dữ liệu đầu vào!")

features_df = pd.DataFrame(features_list)
features_df['Label'] = valid_labels

X = features_df.drop(columns=['Label'])
y = features_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Số lượng mẫu train: {len(X_train)}, Số lượng mẫu test: {len(X_test)}")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
labels = ['Phishing', 'Legitimate']
cm = confusion_matrix(y_test, y_pred, labels=labels)

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

model_path = "../models/random_forest_URL.pkl"
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"Model đã được lưu tại: {model_path}")