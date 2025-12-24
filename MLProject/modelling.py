import pandas as pd
import mlflow
import mlflow.sklearn
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("mlruns")

def train_knn():
    try:
        dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'customer_behavior_preprocessing.csv'
        n_neighbors_val = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except ValueError:
        print("Error: Argumen n_neighbors harus berupa angka.")
        return

    # Mengaktifkan Autologging MLflow
    mlflow.autolog()

    # Menentukan nama eksperimen di MLflow
    mlflow.set_experiment("Eksperimen_SML_KNN_Erza")

    with mlflow.start_run(run_name="KNN_Baseline_Run"):
        try:
            # Load dataset hasil preprocessing
            df = pd.read_csv('Workflow-CI/MLProject/customer_behavior_preprocessing.csv')
            
            # Memisahkan Fitur dan Target
            X = df.drop('Customer_Rating', axis=1)
            y = df['Customer_Rating']
            
            # Split data untuk validasi internal
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            print("🚀 Memulai pelatihan model KNN...")
            
            # Inisialisasi Model KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            
            # Melatih Model
            knn.fit(X_train, y_train)
            
            # Evaluasi
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            
            print(f"Pelatihan Selesai. Accuracy: {acc:.4f}")
            print("\nLaporan Klasifikasi:\n", classification_report(y_val, y_pred))

        except FileNotFoundError:
            print("Error: File 'customer_behavior_preprocessing.csv' tidak ditemukan. Pastikan tahap preprocessing sudah berhasil.")
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    train_knn()
