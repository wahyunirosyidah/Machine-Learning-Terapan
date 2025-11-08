# Customer Churn Prediction using Machine Learning

## Domain Proyek

*Customer churn* merupakan tantangan besar dalam industri telekomunikasi, di mana pelanggan berhenti berlangganan layanan. Tingginya tingkat churn menyebabkan kerugian besar karena biaya untuk mendapatkan pelanggan baru jauh lebih tinggi dibanding mempertahankan pelanggan lama.  

Dengan bantuan *machine learning*, perusahaan dapat **memprediksi pelanggan yang berpotensi churn** berdasarkan data historis. Model prediksi ini dapat membantu perusahaan dalam mengambil langkah pencegahan, seperti promosi atau personalisasi layanan, sebelum pelanggan memutuskan untuk berhenti.  

Dataset yang digunakan adalah **[Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)**, yang berisi data demografis, perilaku pelanggan, dan kontrak langganan.

### Mengapa Masalah Ini Penting

1. **Peningkatan Retensi Pelanggan**  
   Model prediksi churn membantu perusahaan mengidentifikasi pelanggan yang berisiko tinggi berhenti.

2. **Efisiensi Biaya Operasional**  
   Mengurangi churn berarti menghemat biaya pemasaran dan meningkatkan profit.

3. **Pemahaman Pola Pelanggan**  
   Memberikan insight terhadap faktor-faktor yang memengaruhi loyalitas pelanggan seperti harga, kontrak, dan metode pembayaran.

---

## Business Understanding

### Problem Statements
1. Perusahaan belum memiliki sistem yang dapat **memprediksi pelanggan berpotensi churn** secara akurat.  
2. Model prediksi yang ada belum menggabungkan data demografis, perilaku, dan kontrak pelanggan.  
3. Dibutuhkan model dengan **akurasi tinggi dan efisien secara komputasi** agar dapat diimplementasikan secara operasional.

### Goals
1. Mengembangkan model *machine learning classification* untuk memprediksi churn.  
2. Membandingkan performa tiga algoritma: **K-Nearest Neighbors (KNN)**, **Logistic Regression**, dan **XGBoost**.  
3. Menentukan model terbaik berdasarkan *accuracy* dan *F1-score* sebagai metrik utama.

### Solution Statements

- **Solusi 1:**  
  Membangun tiga model klasifikasi menggunakan algoritma:
  - *Logistic Regression*: model linier dan interpretatif (baseline).  
  - *KNN*: model berbasis jarak (*distance-based classifier*).  
  - *XGBoost*: model *ensemble boosting* dengan performa tinggi dan tahan terhadap *overfitting*.

- **Solusi 2:**  
  Mengevaluasi ketiga model menggunakan metrik:
  - **Accuracy**
  - **F1-score**
  - **Classification report** untuk menganalisis performa per kelas (Churn = Yes/No).

---

## Data Understanding

Dataset berisi **7.043 baris dan 21 kolom**, dengan variabel target `Churn` yang menunjukkan apakah pelanggan berhenti (`Yes`) atau tetap (`No`).

### Sumber Data
[Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

### Deskripsi Fitur Utama

| Fitur | Deskripsi |
|-------|------------|
| `customerID` | ID unik pelanggan |
| `gender` | Jenis kelamin pelanggan |
| `SeniorCitizen` | Status lansia |
| `Partner` | Apakah memiliki pasangan |
| `Dependents` | Apakah memiliki tanggungan |
| `tenure` | Lama berlangganan (bulan) |
| `PhoneService` | Layanan telepon |
| `MultipleLines` | Lebih dari satu saluran telepon |
| `InternetService` | Jenis layanan internet |
| `OnlineSecurity`, `OnlineBackup` | Layanan tambahan |
| `DeviceProtection`, `TechSupport` | Proteksi perangkat & dukungan teknis |
| `StreamingTV`, `StreamingMovies` | Layanan hiburan |
| `Contract` | Jenis kontrak pelanggan |
| `PaperlessBilling` | Tagihan digital |
| `PaymentMethod` | Jenis metode pembayaran |
| `MonthlyCharges` | Biaya bulanan |
| `TotalCharges` | Total biaya selama berlangganan |
| `Churn` | Target variabel (Yes = churn, No = stay) |

---

## Exploratory Data Analysis (EDA)

1. **Hubungan Fitur Numerik dengan Churn**
   
   <img width="741" height="741" alt="image" src="https://github.com/user-attachments/assets/27de27ac-1db3-43c9-8df2-83a06df61f48" />
   - `Tenure` berhubungan negatif terhadap churn: semakin lama berlangganan, semakin kecil kemungkinan berhenti.  
   - `MonthlyCharges` berbanding lurus dengan churn: semakin tinggi biaya bulanan, semakin besar kemungkinan churn.  
   - `TotalCharges` menunjukkan pelanggan dengan nilai tinggi lebih loyal.

3. **Contract & Payment Method**

   
   <img width="768" height="432" alt="image" src="https://github.com/user-attachments/assets/4a1f52ab-ccd7-422e-a9a6-fdf96c168e79" />

   - Pelanggan dengan kontrak *Month-to-Month* dan pembayaran manual memiliki tingkat churn tertinggi (47%).  
   - Pelanggan dengan kontrak *Two Year* dan pembayaran otomatis memiliki tingkat churn terendah (sekitar 2–3%).

5. **Distribusi Gender**
   
   <img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/49019ac6-3bee-462f-bc97-927b9dcdbf71" />

   - Jumlah pelanggan pria dan wanita seimbang (Male: 3.549, Female: 3.483), menunjukkan layanan bersifat netral gender.

---

## Data Preparation

1. **Handling Missing Values**  
   Menghapus 11 nilai kosong pada kolom `TotalCharges`.

2. **Encoding Categorical Variables**  
   Menggunakan *One-Hot Encoding* (`pd.get_dummies(drop_first=True)`).

3. **Feature Scaling**  
   Melakukan *standardization* pada fitur numerik (`tenure`, `MonthlyCharges`, `TotalCharges`) dengan **StandardScaler**.

4. **Train-Test Split**  
   Dataset dibagi menjadi 80% data latih dan 20% data uji (`stratify=y`, `random_state=123`).

---

## Modeling

Tiga algoritma *machine learning* digunakan untuk memprediksi churn:

### 1. K-Nearest Neighbors (KNN)
- `n_neighbors=10`
- Kelebihan: sederhana dan intuitif  
- Kekurangan: sensitif terhadap skala dan outlier

### 2. Logistic Regression
- `max_iter=1000`, `solver='liblinear'`
- Cepat, efisien, dan interpretatif — cocok untuk *baseline model*

### 3. XGBoost
- `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `eval_metric='logloss'`
- Kuat terhadap hubungan non-linear dan *imbalanced data*

---

## Evaluation

### Metrik Evaluasi
- **Accuracy**
- **F1-Score**
- **Classification Report**

### Hasil Evaluasi Model

| Model | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|--------|-----------|---------------------|------------------|--------------------|----------------------|------------------|--------------------|
| **KNN** | 0.96 | 0.95 | 0.99 | 0.97 | 0.97 | 0.87 | 0.92 |
| **Logistic Regression** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| **XGBoost** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |


---

### Analisis Hasil

- **XGBoost** menunjukkan performa sangat tinggi dengan **Accuracy = 1.00** dan **F1-score = 1.00**, menandakan model mampu mengenali pola dengan sempurna pada data uji.  
  Namun, hasil yang terlalu sempurna ini dapat mengindikasikan adanya potensi **data leakage** atau **overfitting**, karena sangat jarang model mencapai akurasi 100% pada data dunia nyata.

- **Logistic Regression** juga memperoleh hasil identik (**Accuracy = 1.00**, **F1-score = 1.00**).  
  Model ini tetap kompetitif dan menjadi **baseline model** yang sangat baik karena sederhana, stabil, dan mudah diinterpretasikan.

- **K-Nearest Neighbors (KNN)** menghasilkan performa **Accuracy = 0.96** dan **F1-score = 0.94**, yang lebih realistis dibanding dua model lainnya.  
  KNN cukup baik dalam mengenali pola pelanggan churn, tetapi sensitif terhadap *noise* dan membutuhkan waktu komputasi lebih tinggi pada dataset besar.

Secara keseluruhan, meskipun **XGBoost** dan **Logistic Regression** menunjukkan hasil sempurna, diperlukan **validasi lanjutan** seperti *K-Fold Cross Validation* untuk memastikan model benar-benar mampu melakukan generalisasi terhadap data baru.

---

## Keterkaitan dengan Business Understanding

| Problem | Status | Penjelasan |
|----------|---------|------------|
| Tidak ada model prediksi churn akurat | ✅ Terjawab | Model **XGBoost** dan **Logistic Regression** mencapai akurasi tinggi dan konsisten |
| Faktor penyebab churn belum jelas | ✅ Terjawab | Analisis menunjukkan fitur **Contract**, **Tenure**, dan **MonthlyCharges** paling berpengaruh terhadap churn |
| Perlu model efisien untuk operasional | ✅ Terjawab | **Logistic Regression** efisien secara komputasi dan mudah diimplementasikan di lingkungan bisnis |

---

## Kesimpulan

- **Model terbaik:** XGBoost — memiliki performa tertinggi dan mampu menangkap hubungan non-linear antar fitur dengan sangat baik.  
- **Baseline model:** Logistic Regression — akurat, stabil, dan interpretatif sehingga cocok untuk implementasi awal.  
- **KNN** digunakan sebagai pembanding sederhana, namun kurang efisien dan sedikit lebih rendah performanya dibanding dua model lainnya.

Model prediksi ini dapat membantu perusahaan untuk:
- Mengidentifikasi pelanggan dengan risiko tinggi melakukan churn.  
- Menyusun strategi retensi pelanggan yang lebih terarah dan berbasis data.  
- Meningkatkan efisiensi biaya serta memperkuat loyalitas pelanggan.

---

## Referensi

1. [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
2. Han, J., Kamber, M., & Pei, J. (2022). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.  
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. *Proceedings of the 22nd ACM SIGKDD*.  
