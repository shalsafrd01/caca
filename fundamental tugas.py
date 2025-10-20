import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score,precision_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('TUGASSS/katalog_gempa.csv')
data.info()
data.describe()

 # Memeriksa apakah ada nilai dalam dataset yang "kosong" atau "NaN"
print(data.isnull().values.any())
print(data.isnull().sum())
data = data.fillna(data.mean(numeric_only=True))
print(data.isnull().sum())
print(data.columns)

# ===================================================
#  Memilih Fitur (Features) dan Target
# ===================================================

# Misal kita ingin memprediksi kolom 'Activity' berdasarkan sensor data lainnya
# (ganti nama kolom sesuai dataset kamu)
target_column = 'tgl'  # kolom target
feature_columns = ['ot', 'lat', 'lon', 'depth', 'mag', 'remark', 'strike1', 'dip1','rake1', 'strike2', 'dip2', 'rake2']  # fitur = semua kolom lain

# Pisahkan fitur dan target
X = data[feature_columns]
y = data[target_column]

print("tgl")
print(feature_columns)
print("\TUGASSS/katalog_gempa.csv")
print(y.head())

print(data.dtypes)
# Hapus kolom non-numerik seperti tanggal dan teks
data_numeric = data.select_dtypes(include=['int64', 'float64'])

# Tentukan fitur dan target
target_column = 'mag'  # contoh target numerik (ubah sesuai kebutuhan)
feature_columns = [col for col in data_numeric.columns if col != target_column]

X = data_numeric[feature_columns]
y = data_numeric[target_column]

# Discretizer
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(
    n_bins=4,
    encode='ordinal',
    strategy='quantile',
    quantile_method='averaged_inverted_cdf'
)
y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
print("Data target setelah diskritisasi:")
print(y_binned[:10])

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X,
y_binned, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_train:", y_train.shape)
print("Shape of Y_test:", y_test.shape)


import numpy as np
from sklearn.metrics import accuracy_score
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# 2. Mencari kelas mayoritas
majority_class = np.bincount(y_train).argmax()
print(f"Kelas mayoritas adalah: {majority_class}")

# 3. Membuat prediksi baseline (semua data diprediksi ke kelas mayoritas)
y_pred_baseline = np.full_like(y_test, majority_class)

# 4. Menghitung akurasi baseline
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Akurasi baseline: {baseline_accuracy:.2f}")

# Inisialisasi model
model = DecisionTreeClassifier(random_state=42)
 # Latih model menggunakan data latih
model.fit(X_train, y_train)
 # Buat prediksi menggunakan data uji
y_pred = model.predict(X_test)

 # Hitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
labels = ['Low', 'Medium', 'High','Very High']
 # Menghitung akurasi baseline
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print("Baseline Accuracy:", baseline_accuracy,"\n")
print("Decicion Tree Accuracy:", accuracy)
print("Decision Tree Precision:", precision, "\n")
 # Tampilkan confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
 # Tampilkan classification report
print(classification_report(y_test, y_pred))
