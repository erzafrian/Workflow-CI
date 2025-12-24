import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import argparse # More robust than sys.argv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

mlflow.set_tracking_uri("mlruns")

def train_and_log(n_neighbors, weights):
    mlflow.set_experiment("KNN_Tuning_Final")

    try:
        # 1. Load Dataset
        csv_path = './customer_behavior_preprocessing.csv'
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")

        df = pd.read_csv(csv_path)
        X = df.drop('Customer_Rating', axis=1)
        y = df['Customer_Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Start MLflow Run
        with mlflow.start_run(run_name=f"KNN_k{n_neighbors}_{weights}"):
            print(f"Training model: n_neighbors={n_neighbors}, weights={weights}")
            
            # Use the parameters passed from CLI
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Log Params and Metrics
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("weights", weights)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            mlflow.log_metrics(metrics)

            # 3. Model Signature & Artifacts
            signature = infer_signature(X_train, knn.predict(X_train))
            local_dir = "temp_artifacts"
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            os.makedirs(local_dir)

            # Save model locally first to bundle it
            model_path = os.path.join(local_dir, "knn_model")
            mlflow.sklearn.save_model(knn, model_path, input_example=X_train[:5], signature=signature)

            # Confusion Matrix
            plt.figure(figsize=(8,6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
            plt.title(f'Confusion Matrix (k={n_neighbors})')
            plot_path = os.path.join(local_dir, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()

            # Classification Report
            report_path = os.path.join(local_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))

            # Upload entire folder as artifact
            mlflow.log_artifacts(local_dir, artifact_path="model_output")
            
            print(f"Selesai! Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, required=True, default=5)
    parser.add_argument("--weights", type=int, required=True, default=3)
    args = parser.parse_args()
    train_and_log(args.n_neighbors, args.weights)
