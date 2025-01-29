import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["nameOrig", "nameDest"])
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])
    return df

def preprocess_features(df):
    selected_features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df

def train_model(X_scaled, df, contamination_rate=0.001):
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(X_scaled)
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["isFraud_pred"] = model.predict(X_scaled)
    df["isFraud_pred"] = df["isFraud_pred"].apply(lambda x: 1 if x == -1 else 0)
    return model, df

def evaluate_model(df):
    print(classification_report(df["isFraud"], df["isFraud_pred"]))
    conf_matrix = confusion_matrix(df["isFraud"], df["isFraud_pred"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fraud Detection Confusion Matrix")
    plt.show()

def main():
    file_path = r"C:\Users\ipek\Downloads\PS_20174392719_1491204439457_log.csv"
    df = load_and_clean_data(file_path)
    X_scaled, df = preprocess_features(df)
    model, df = train_model(X_scaled, df)
    evaluate_model(df)

if __name__ == "__main__":
    main()
