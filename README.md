# Workflow CI/CD MLflow – Fraud Detection Model

Repository ini berisi implementasi **Workflow CI/CD Machine Learning** menggunakan **MLflow Project** dan **GitHub Actions**.  
Workflow ini akan melakukan **training model secara otomatis**, menyimpan artefak MLflow, serta **membangun dan mendorong Docker Image ke Docker Hub** setiap kali terjadi trigger.

---

## Struktur Repository
Workflow-CI_Rindumas-Ismara-Putri
├── .github
│ └── workflows
│ └── main_ci.yaml
├── MLProject
│ ├── MLProject
│ ├── modelling.py
│ ├── conda.yaml
│ └── fraud_detection_preprocessing.csv
├── requirements.txt
├── trigger.txt
└── README.md

---

## Deskripsi Workflow

Workflow CI ini dirancang untuk memastikan berjalannya yang sudah dilakukan pada eksperimen sebelumnya berhasil, dengan tahapan:

1. **Trigger Workflow**
   - Workflow dijalankan saat:
     - Push ke branch `main`
     - Manual trigger (`workflow_dispatch`)

2. **Training Model**
   - Menggunakan **MLflow Project**
   - Parameter model didefinisikan di file `MLProject`
   - Script utama: `modelling.py`
   - Algoritma: **RandomForestClassifier**

3. **Logging & Artifacts**
   - Parameter dan metrik dicatat menggunakan MLflow
   - Artefak yang disimpan:
     - Model ML
     - Confusion Matrix
     - Feature Importance
   - Artefak diunggah sebagai **GitHub Actions Artifact**

4. **Docker Image**
   - Model MLflow dikemas menjadi Docker Image menggunakan:
     ```bash
     mlflow models build-docker
     ```
   - Image menggunakan **MLServer** untuk inference
   - Image otomatis di-push ke Docker Hub

---

## Docker Hub Repository

Docker Image hasil training dapat diakses di:

**Docker Hub - Fraud Detection Model** : https://hub.docker.com/repository/docker/rindumasismara/fraud-detection-model/general

Tag yang tersedia:
- `latest`
- `<RUN_ID>` (berdasarkan MLflow Run ID)

---

## Teknologi yang Digunakan

- Python 3.12.7
- MLflow 2.19.0
- Scikit-learn
- Pandas & NumPy
- GitHub Actions
- Docker & Docker Hub
- MLServer

---

## ▶ Cara Menjalankan Workflow

Workflow akan berjalan otomatis saat:
- Melakukan `git push` ke branch `main`, atau
- Menjalankan manual dari tab **Actions** di GitHub

Tidak diperlukan menjalankan script secara manual di lokal.

---

## Pemilik

**Rindumas Ismara Putri**  
Eksperimen Sistem Machine Learning