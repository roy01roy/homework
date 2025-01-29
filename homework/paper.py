import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_and_clean_data(file_path):
    print("Data loading...")
    df = pd.read_csv(file_path)
    print(f"Data loaded! {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Cleaning data...")
    df['is_type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    df['hour'] = df['step'] % 24
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    df.drop(['step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud', 'hour'], axis=1, inplace=True)
    print("Data cleaning complete.")

    print("Scaling numerical features...")
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'sin_hour', 'cos_hour']
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    print("Scaling complete.")

    return df

def train_and_evaluate_models(df):
    print("\nSplitting data into train and test sets...")
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows.")

    print("\nApplying SMOTE to balance data...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: Train set now has {X_train.shape[0]} rows.")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    print("\nTraining models...")
    for name, model in tqdm(models.items(), desc="Model Training"):
        print(f"\nTraining {name} model...")
        model.fit(X_train, y_train)
        print(f"{name} model trained.")

        print(f"\nTesting {name} model...")
        y_pred = model.predict(X_test)

        print(f"\n{name} - Performance Report:")
        results[name] = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)

    return results

def plot_confusion_matrix(cm, model_name):
    print(f"\nDrawing Confusion Matrix for {model_name}...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    print(f"Confusion Matrix drawn for {model_name}.")

def main():
    file_path = "C:\\Users\\ipek\\Downloads\\PS_20174392719_1491204439457_log.csv"
    df = load_and_clean_data(file_path)
    results = train_and_evaluate_models(df)
    return results

if __name__ == "__main__":
    print("Program started!")
    main()
