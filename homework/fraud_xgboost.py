import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["nameOrig", "nameDest"])
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])
    return df

def preprocess_features(df):
    df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["transaction_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    selected_features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                         "balance_diff", "transaction_ratio"]
    X = df[selected_features]
    y = df["isFraud"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def balance_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

def evaluate_model(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fraud Detection Confusion Matrix (XGBoost)")
    plt.show()

def main():
    file_path = r"C:\Users\ipek\Downloads\PS_20174392719_1491204439457_log.csv"
    df = load_and_clean_data(file_path)
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = balance_data(X, y)
    model, y_pred, y_test = train_model(X_train, X_test, y_train, y_test)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
