import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from pre_process import load_data, preprocess_data

CSV_PATH = "../dataset_URL/phishing_URL.csv"
MODEL_PATH = "../models/random_forest_URL.pkl"
RESULTS_PATH = "../results/evaluation.txt"

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    plt.close()

def save_evaluation(cm, accuracy, precision, recall, report):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("Ma trận nhầm lẫn (labels=[1: Phishing, 0: Benign]):\n")
        f.write(f"{cm}\n\n")
        f.write("Đánh giá mô hình:\n")
        f.write(f"- Accuracy  : {accuracy:.2f}\n")
        f.write(f"- Precision : {precision:.2f}\n")
        f.write(f"- Recall    : {recall:.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(f"{report}\n")
    print(f"Kết quả đánh giá đã được lưu tại: {RESULTS_PATH}")

def rf_model():
    df = load_data(CSV_PATH)

    X, y = preprocess_data(df)
    
    expected_features = len(df.columns) - 2
    if X.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"+ Số lượng mẫu train: {len(X_train)}")
    print(f"+ Số lượng mẫu test: {len(X_test)}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    if not isinstance(rf, RandomForestClassifier):
        raise TypeError(f"Model is not a RandomForestClassifier, got {type(rf)}")

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"Mô hình đã được lưu tại: {MODEL_PATH}")

    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    print("\nMa trận nhầm lẫn (labels=[1: Phishing, 0: Benign]):")
    print(cm)

    tp, fn = cm[0]
    fp, tn = cm[1]
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    report = classification_report(y_test, y_pred, target_names=["0 (Benign)", "1 (Phishing)"])

    print("\nĐánh giá mô hình:")
    print(f"- Accuracy  : {accuracy:.2f}")
    print(f"- Precision : {precision:.2f}")
    print(f"- Recall    : {recall:.2f}")
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, labels=[1, 0]) 
    save_evaluation(cm, accuracy, precision, recall, report)

if __name__ == "__main__":
    rf_model()