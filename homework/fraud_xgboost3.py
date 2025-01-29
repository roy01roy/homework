import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix


# 📌 1. Veri Yükleme ve Temizleme
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["nameOrig", "nameDest"])  # Kimlik bilgilerini kaldır
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])  # Kategorik değişkenleri sayısala çevir
    print("✅ Veri başarıyla yüklendi ve temizlendi!")
    return df


# 📌 2. Özellik Seçimi ve Yeni Feature Ekleyelim
def preprocess_features(df):
    df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["transaction_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    selected_features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                         "balance_diff", "transaction_ratio"]
    X = df[selected_features]
    y = df["isFraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Veri ölçeklendi ve modele hazır!")
    return X_scaled, y


# 📌 3. LOF ile Fraud İşlemleri Anomali Olarak İşaretleme
def detect_anomalies(X, y):
    """Local Outlier Factor (LOF) kullanarak fraud işlemleri belirgin hale getirelim."""
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    anomaly_scores = lof.fit_predict(X)

    # LOF -1 olarak anomalileri işaretler, bunu fraud işlemleriyle eşleştiriyoruz
    y_lof = np.where(anomaly_scores == -1, 1, y)

    print(f"✅ LOF ile {sum(y_lof)} fraud işlemi işaretlendi (Önce: {sum(y)} idi).")
    return y_lof


# 📌 4. Cost-Sensitive Learning ile Ağırlıklandırma
def balance_data(X, y):
    """Veri setinde fraud işlemleri az olduğu için ağırlıklandırma uygulayacağız."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fraud işlemleri için ağırlık hesapla
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    print(f"✅ Veri başarıyla ayrıldı: Fraud işlemler oranı = {sum(y_train) / len(y_train):.4f}")
    return X_train, X_test, y_train, y_test, sample_weights


# 📌 5. Cost-Sensitive XGBoost Modelini Eğitme
def train_model(X_train, X_test, y_train, y_test, sample_weights):
    """Fraud işlemlerini yakalamak için Cost-Sensitive Learning kullanacağız."""
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # Fraud işlemleri daha önemli hale getiriyoruz!
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    print("✅ Model eğitildi ve tahminler yapıldı!")
    return model, y_pred, y_test


# 📌 6. Modeli Değerlendirme
def evaluate_model(y_test, y_pred):
    print("\n🔍 Model Performansı:\n")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Fraud", "Fraud"],
                yticklabels=["No Fraud", "Fraud"])
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Fraud Tespiti İçin Confusion Matrix (LOF + Cost-Sensitive XGBoost)")
    plt.show()


# 📌 7. Ana Çalıştırma Fonksiyonu
def main():
    file_path = r"C:\Users\ipek\Downloads\PS_20174392719_1491204439457_log.csv"  # Dosya yolunu güncelle

    df = load_and_clean_data(file_path)
    X, y = preprocess_features(df)
    y_lof = detect_anomalies(X, y)  # LOF ile fraud işlemleri belirgin hale getiriyoruz
    X_train, X_test, y_train, y_test, sample_weights = balance_data(X, y_lof)
    model, y_pred, y_test = train_model(X_train, X_test, y_train, y_test, sample_weights)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()
