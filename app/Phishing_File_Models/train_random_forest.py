import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from preprocess import preprocess_data, load_data

CSV_PATH = "../dataset_File/data_File.csv"

def rf_model():
    df = load_data(CSV_PATH)
    print("- Thống kê file Benign:")
    print(df[df['Class'].str.lower() == 'benign'].describe())
    
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"+ Số lượng mẫu train: {len(X_train)}")
    print(f"+ Số lượng mẫu test: {len(X_test)}")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "../models/random_forest_file.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Model đã được lưu tại: {model_path}")

    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision_manual = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) != 0 else 0

    print("\nĐánh giá mô hình:")
    print(f"- Độ chính xác (Accuracy): {accuracy_manual:.2f}")
    print(f"- Độ chính xác dự đoán Phishing (Precision): {precision_manual:.2f}")
    print(f"- Khả năng nhận diện Phishing đúng (Recall): {recall_manual:.2f}")

if __name__ == "__main__":
    rf_model()