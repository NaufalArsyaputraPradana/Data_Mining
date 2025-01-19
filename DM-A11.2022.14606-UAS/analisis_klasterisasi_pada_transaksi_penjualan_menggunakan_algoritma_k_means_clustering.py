from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
import pandas as pd  # Untuk manipulasi dan analisis data
from sklearn.cluster import KMeans  # Algoritma KMeans untuk clustering
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi data
# Untuk encoding variabel kategorikal
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Untuk visualisasi data dan hasil model
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
import numpy as np  # Untuk manipulasi array dan operasi numerik
# %matplotlib inline

# Judul aplikasi
st.title("Analisis Klasterisasi pada Transaksi Penjualan")

# Mengunggah file CSV
df = pd.read_csv('DM-A11.2022.14606-UAS/barang_keluar.csv',
                 delimiter=';', skiprows=0, low_memory=False)

# Menampilkan beberapa baris pertama dari dataset
st.write("Data yang dimuat:")
st.dataframe(df.head())

# Memeriksa informasi dataset
st.write("Informasi Dataset:")
st.dataframe(df)  # Directly display the DataFrame without styling

# Memeriksa statistik deskriptif
st.write("Statistik Deskriptif:")
st.dataframe(df.describe())

# Memilih kolom yang relevan untuk clustering
df_selected = df[['jumlah_transaksi', 'total_penjualan']]
st.write("Data yang dipilih untuk clustering:")
st.dataframe(df_selected.head())

# Visualisasi hubungan antara jumlah_transaksi dan total_penjualan
plt.figure(figsize=(8, 6))
plt.scatter(df_selected['jumlah_transaksi'],
            df_selected['total_penjualan'], c='blue', alpha=0.5)
plt.title('Hubungan antara Jumlah Transaksi dan Total Penjualan')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.grid(True)
st.pyplot(plt)

"""# **Pra Proses Data (Cleansing)**

Memeriksa Nilai yang Hilang
"""

# Memeriksa jumlah nilai yang hilang (NaN) dalam setiap kolom
st.write("Missing values:\n", df.isnull().sum())

# Memeriksa persentase nilai yang hilang
st.write("\nPercentage of Missing Values:\n", df.isnull().mean() * 100)

"""Menangani Nilai yang Hilang (Missing Values)"""

# Menghapus baris yang mengandung nilai kosong (NaN)
df_cleaned = df.dropna()

# Memeriksa apakah nilai kosong sudah hilang setelah pembersihan
st.write("\nMissing values after cleaning:\n", df_cleaned.isnull().sum())

"""Menangani Duplikat"""

# Memeriksa jumlah duplikat dalam dataset
st.write("\nJumlah duplikat:", df_cleaned.duplicated().sum())

# Menghapus duplikat
df_cleaned = df_cleaned.drop_duplicates()

# Memeriksa ulang apakah duplikat sudah dihapus
st.write("\nJumlah duplikat setelah pembersihan:", df_cleaned.duplicated().sum())

"""Memeriksa dan Menangani Outliers"""

# Visualisasi menggunakan boxplot untuk memeriksa outliers pada kolom 'jumlah_transaksi' dan 'total_penjualan'
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned[['jumlah_transaksi', 'total_penjualan']])
plt.title('Boxplot Jumlah Transaksi dan Total Penjualan')
st.pyplot(plt)

"""# **Pra Proses Data (Transform)**

Normalisasi Data (Scaling)
"""

# Inisialisasi objek MinMaxScaler untuk normalisasi data
scaler = MinMaxScaler()

# Menormalkan kolom 'total_penjualan' dengan MinMaxScaler
df['total_penjualan_cluster'] = scaler.fit_transform(df[['total_penjualan']])

# Menormalkan kolom 'jumlah_transaksi' dengan MinMaxScaler
df['jumlah_transaksi_cluster'] = scaler.fit_transform(df[['jumlah_transaksi']])

# Jika ada fitur lain yang perlu dinormalisasi (misalnya 'rata_rata')
# df['rata_rata_cluster'] = scaler.fit_transform(df[['rata_rata']])

# Tampilkan beberapa data pertama setelah normalisasi
st.dataframe(df.head())

# Visualisasi distribusi data sebelum dan sesudah normalisasi
plt.figure(figsize=(14, 6))

# Distribusi sebelum normalisasi
plt.subplot(1, 2, 1)
plt.hist(df['total_penjualan'], bins=30, color='blue',
         alpha=0.7, label='Total Penjualan (Sebelum)')
plt.hist(df['jumlah_transaksi'], bins=30, color='green',
         alpha=0.7, label='Jumlah Transaksi (Sebelum)')
plt.title('Distribusi Sebelum Normalisasi')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.legend()

# Distribusi setelah normalisasi
plt.subplot(1, 2, 2)
plt.hist(df['total_penjualan_cluster'], bins=30, color='blue',
         alpha=0.7, label='Total Penjualan (Sesudah)')
plt.hist(df['jumlah_transaksi_cluster'], bins=30, color='green',
         alpha=0.7, label='Jumlah Transaksi (Sesudah)')
plt.title('Distribusi Setelah Normalisasi')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.legend()

plt.tight_layout()
st.pyplot(plt)

# Menyaring kolom yang akan digunakan dalam analisis
df_scaled = df_cleaned[['jumlah_transaksi', 'total_penjualan']]

# Menggunakan MinMaxScaler untuk menormalisasi data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_scaled)

# Menampilkan hasil transformasi (scaled values)
# Menampilkan 5 baris pertama dari data yang sudah dinormalisasi
st.write("Data setelah normalisasi:\n", df_scaled[:5])

"""Encoding Kategorikal Data (Label Encoding)"""

# Menggunakan LabelEncoder untuk mengubah kolom kategorikal menjadi numerik
encoder = LabelEncoder()
df_cleaned['kode_barang_encoded'] = encoder.fit_transform(
    df_cleaned['kode_barang'])

# Menampilkan 5 baris pertama untuk melihat hasil encoding
st.write("\nData setelah Label Encoding:\n", df_cleaned[[
      'kode_barang', 'kode_barang_encoded']].head())

"""Menggabungkan Data yang Ditransformasi"""

# Menggabungkan data yang sudah di-normalisasi dengan data yang sudah di-encode
df_transformed = pd.DataFrame(
    df_scaled, columns=['jumlah_transaksi_scaled', 'total_penjualan_scaled'])
df_transformed['kode_barang_encoded'] = df_cleaned['kode_barang_encoded']

# Menampilkan hasil akhir setelah transformasi
st.write("\nData setelah transformasi:\n", df_transformed.head())

"""# **Modelling Clustering**"""

km = KMeans(n_clusters=3)
km

"""Clusters ada 3 :
1. Data Penjualan Rendah
2. Data Penjualan Sedang
3. Data Penjualan Tinggi
"""

# Menentukan jumlah cluster optimal menggunakan Elbow Method
inertia = []
for k in range(1, 11):  # Menguji cluster antara 1 hingga 10
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df[['jumlah_transaksi', 'total_penjualan']])
    inertia.append(km.inertia_)

# Plot Elbow Method
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
st.pyplot(plt)

inertia = []
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, km.labels_))

# Plot Elbow Method dan Silhouette Score
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
st.pyplot(plt)

"""# **Hasil Clustering**"""

# Menambahkan 2 karena dimulai dari 2
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
km = KMeans(n_clusters=optimal_k, random_state=42)
y_predicted = km.fit_predict(df[['jumlah_transaksi', 'total_penjualan']])

# Menampilkan hasil cluster yang diprediksi
st.write("Cluster yang diprediksi untuk setiap data:\n", y_predicted[:10])  # Menampilkan 10 data pertama

# Melakukan K-Means Clustering dengan jumlah cluster K=3
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['jumlah_transaksi', 'total_penjualan']])

# Menampilkan hasil cluster yang diprediksi
st.write("Cluster yang diprediksi untuk setiap data:\n", y_predicted[:10])  # Menampilkan 10 data pertama

"""Menambahkan Hasil Prediksi Cluster ke Data"""

y_predicted = km.fit_predict(df[['jumlah_transaksi', 'total_penjualan']])
y_predicted

# Menambahkan hasil prediksi cluster ke dataframe
df['cluster'] = y_predicted

# Menampilkan 5 baris pertama untuk memastikan kolom 'cluster' ditambahkan dengan benar
st.write("Data dengan kolom cluster:\n", df.head())

# Menampilkan distribusi jumlah data pada setiap cluster
cluster_distribution = df['cluster'].value_counts()
st.write("Distribusi jumlah data pada setiap cluster:\n", cluster_distribution)

# Inisialisasi model K-Means dengan jumlah cluster (n_clusters) = 3
km = KMeans(n_clusters=3, random_state=42,
            init='k-means++', max_iter=300, n_init=10)

# Melakukan clustering pada kolom 'jumlah_transaksi' dan 'total_penjualan'
y_predicted = km.fit_predict(df[['jumlah_transaksi', 'total_penjualan']])

# Menambahkan hasil prediksi cluster ke dalam dataset
df['cluster'] = y_predicted

# Evaluasi menggunakan Silhouette Score untuk melihat kualitas clustering
silhouette_avg = silhouette_score(
    df[['jumlah_transaksi', 'total_penjualan']], y_predicted)

# Menampilkan hasil clustering dan evaluasi
# Menampilkan 10 hasil pertama
st.write(f"Hasil Prediksi Cluster: {y_predicted[:10]}")
# Menampilkan Silhouette Score
st.write(f"Silhouette Score: {silhouette_avg:.3f}")
df.head()

"""# **Visualisasi Hasil Clustering**"""

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
plt.scatter(df['jumlah_transaksi'], df['total_penjualan'],
            c=df['cluster'], cmap='viridis')
plt.title('Hasil Clustering K-Means')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Memisahkan data berdasarkan cluster
df1 = df[df.cluster == 0]  # Cluster 0
df2 = df[df.cluster == 1]  # Cluster 1
df3 = df[df.cluster == 2]  # Cluster 2
# df4 = df[df.cluster == 3]  # Cluster 3, jika ada lebih dari 3 cluster

# Visualisasi hasil clustering dengan scatter plot
plt.figure(figsize=(10, 6))

# Plot untuk Cluster 0 (warna hijau)
plt.scatter(df1['jumlah_transaksi'], df1['total_penjualan'],
            color='green', label='Cluster 0')

# Plot untuk Cluster 1 (warna merah)
plt.scatter(df2['jumlah_transaksi'], df2['total_penjualan'],
            color='red', label='Cluster 1')

# Plot untuk Cluster 2 (warna biru)
plt.scatter(df3['jumlah_transaksi'], df3['total_penjualan'],
            color='blue', label='Cluster 2')

# Jika ada cluster 4, aktifkan kode berikut
# plt.scatter(df4['jumlah_transaksi'], df4['total_penjualan'], color='gray', label='Cluster 3')

# Menambahkan label dan judul pada plot
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.title(
    'Visualisasi Hasil Clustering Berdasarkan Jumlah Transaksi dan Total Penjualan')

# Menambahkan legend untuk setiap cluster
plt.legend()

# Menampilkan plot
st.pyplot(plt)

"""Menampilkan Centroid dari Setiap Cluster"""

# Menampilkan posisi centroid
centroids = km.cluster_centers_

# Visualisasi centroid
plt.scatter(df['jumlah_transaksi'], df['total_penjualan'],
            c=df['cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200,
            c='red', marker='X', label='Centroid')
plt.title('Hasil Clustering dengan Centroid')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
plt.legend()
st.pyplot(plt)

# Analisis statistik dari tiap cluster
cluster_analysis = df.groupby('cluster').agg({
    'jumlah_transaksi': ['mean', 'std', 'min', 'max'],
    'total_penjualan': ['mean', 'std', 'min', 'max']
})

st.write("\nAnalisis Statistik Tiap Cluster:\n", cluster_analysis)

"""# **Centroid Clustering**"""

km.cluster_centers_

# Menampilkan pusat cluster (centroid)
centroids = km.cluster_centers_

# Menyusun output dalam format tabel untuk kejelasan
centroids_df = pd.DataFrame(
    centroids, columns=['Jumlah Transaksi', 'Total Penjualan'])

# Menampilkan centroid dengan lebih informatif
st.write("Pusat Cluster (Centroids):\n", centroids_df)

# Visualisasi posisi centroid
plt.figure(figsize=(8, 6))
plt.scatter(df['jumlah_transaksi'], df['total_penjualan'],
            c=y_predicted, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300,
            c='red', marker='X', label='Centroids')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.title('Cluster and Centroids Visualization')
plt.legend()
st.pyplot(plt)

km.labels_

# Mendapatkan label cluster untuk setiap data poin
labels = km.labels_

# Menambahkan label cluster ke DataFrame asli
df['Cluster'] = labels

# Menampilkan 10 data pertama beserta label clusternya
st.write("Data dengan Label Cluster:\n", df.head(10))

# Menghitung distribusi data dalam setiap cluster
cluster_distribution = df['Cluster'].value_counts().sort_index()

st.write("\nDistribusi Data dalam Setiap Cluster:")
for cluster, count in cluster_distribution.items():
    st.write(f"Cluster {cluster}: {count} data points")

# Visualisasi distribusi data dalam setiap cluster
plt.figure(figsize=(8, 5))
plt.bar(cluster_distribution.index, cluster_distribution.values,
        color='skyblue', edgecolor='black')
plt.title('Distribusi Data dalam Setiap Cluster')
plt.xlabel('Cluster')
plt.ylabel('Jumlah Data')
plt.xticks(ticks=cluster_distribution.index)
st.pyplot(plt)

# Pisahkan data berdasarkan cluster
df_cluster_0 = df[df['Cluster'] == 0]
df_cluster_1 = df[df['Cluster'] == 1]
df_cluster_2 = df[df['Cluster'] == 2]

# Visualisasi data dalam setiap cluster
plt.figure(figsize=(10, 6))
plt.scatter(
    df_cluster_0['jumlah_transaksi'], df_cluster_0['total_penjualan'],
    color='green', label='Cluster 0', alpha=0.7, edgecolor='black'
)
plt.scatter(
    df_cluster_1['jumlah_transaksi'], df_cluster_1['total_penjualan'],
    color='red', label='Cluster 1', alpha=0.7, edgecolor='black'
)
plt.scatter(
    df_cluster_2['jumlah_transaksi'], df_cluster_2['total_penjualan'],
    color='blue', label='Cluster 2', alpha=0.7, edgecolor='black'
)

# Plotkan centroid
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    color='gold', marker='*', label='Centroid', s=200, edgecolor='black'
)

# Menambahkan judul dan label
plt.title('Visualisasi Clustering dengan K-Means', fontsize=16)
plt.xlabel('Jumlah Transaksi', fontsize=12)
plt.ylabel('Total Penjualan', fontsize=12)

# Menambahkan legenda
plt.legend(loc='upper right', fontsize=10)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Menampilkan grafik
st.pyplot(plt)

k_rng = range(1, 10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['jumlah_transaksi', 'total_penjualan']])
    sse.append(km.inertia_)

sse


# Inisialisasi rentang jumlah cluster
k_range = range(1, 10)
sse = []  # List untuk menyimpan nilai SSE

# Iterasi untuk menghitung SSE pada tiap nilai k
for k in k_range:
    # Tambahkan random_state untuk hasil konsisten
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['jumlah_transaksi', 'total_penjualan']])  # Fitting data
    sse.append(kmeans.inertia_)  # Simpan nilai inertia (SSE)

# Visualisasi SSE vs Jumlah Cluster
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o', linestyle='-', color='blue')

# Menambahkan detail grafik
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal', fontsize=14)
plt.xlabel('Jumlah Cluster (k)', fontsize=12)
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
plt.xticks(k_range)  # Pastikan semua nilai k ditampilkan di sumbu X
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Menampilkan grafik
st.pyplot(plt)

# Menampilkan hasil SSE untuk setiap jumlah cluster
for k, error in zip(k_range, sse):
    st.write(f"Jumlah Cluster (k): {k}, SSE: {error:.2f}")

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_rng, sse)

"""# **Evaluasi Data**"""

# Seleksi kolom fitur yang relevan
selected_cols = ["jumlah_transaksi", "total_penjualan"]
# Mengambil kolom yang akan digunakan untuk clustering
cluster_data = df[selected_cols]

# Inisialisasi model K-Means dengan parameter yang dioptimalkan
kmeans_sel = KMeans(
    init='k-means++',        # Metode pengaturan awal centroid yang lebih optimal
    n_clusters=3,            # Jumlah cluster yang telah ditentukan
    n_init=100,              # Jumlah inisialisasi ulang untuk mendapatkan hasil terbaik
    random_state=42          # Random state untuk hasil yang konsisten
)

# Fitting model K-Means ke data cluster yang dipilih
kmeans_sel.fit(cluster_data)

# Menambahkan label cluster ke dataset
cluster_data['Cluster'] = kmeans_sel.labels_

# Menampilkan beberapa baris pertama dari dataset yang sudah diberi label cluster
st.write(cluster_data.head())

# Mengelompokkan data berdasarkan cluster dan menghitung rata-rata untuk setiap fitur
grouped_km = cluster_data.groupby('Cluster').mean().round(1)

# Menampilkan hasil pengelompokan
st.write("Rata-rata fitur dalam setiap cluster:")
st.write(grouped_km)

print(confusion_matrix(df['cluster'], km.labels_))


# Pastikan Anda memiliki label referensi (misalnya: df['true_label'])
if 'true_label' in df.columns:  # Hanya jika true_label tersedia
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(df['true_label'], km.labels_))
    st.write("\nClassification Report:")
    st.write(classification_report(df['true_label'], km.labels_))
else:
    st.write("Label referensi (true_label) tidak ditemukan dalam dataset. Evaluasi clustering hanya berdasarkan cluster.")


# Menghitung skor evaluasi clustering
sil_score = silhouette_score(
    df[['jumlah_transaksi', 'total_penjualan']], km.labels_)
ch_score = calinski_harabasz_score(
    df[['jumlah_transaksi', 'total_penjualan']], km.labels_)

st.write(f"Silhouette Score: {sil_score:.3f}")
st.write(f"Calinski-Harabasz Index: {ch_score:.3f}")


# Menghitung korelasi antar fitur
correlation_matrix = df[['jumlah_transaksi', 'total_penjualan']].corr()

# Plot heatmap untuk korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)


# Menampilkan classification report untuk membandingkan label asli dengan label hasil clustering
st.write(classification_report(df['cluster'], km.labels_))

df.sort_values(by=['cluster'])

# Mengurutkan DataFrame berdasarkan kolom 'cluster'
df_sorted = df.sort_values(by=['cluster'])

# Menampilkan data yang sudah diurutkan
st.write(df_sorted.head())

# Menetapkan opsi untuk menampilkan semua baris dalam DataFrame
pd.set_option('display.max_rows', df.shape[0] + 1)

# Mengurutkan DataFrame berdasarkan kolom 'cluster'
df_sorted = df.sort_values(by=['cluster'])

st.write(df.head())

df.to_csv('datasets-jadi.csv', index=False)

"""# **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**"""


# Menggunakan StandardScaler untuk melakukan normalisasi data (karena DBSCAN sensitif terhadap skala data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    df_cleaned[['jumlah_transaksi', 'total_penjualan']])

# Inisialisasi DBSCAN
# eps menentukan jarak maksimum antar titik untuk membentuk sebuah cluster
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Menambahkan hasil clustering DBSCAN ke dalam DataFrame
df_cleaned['dbscan_cluster'] = dbscan_labels

# Visualisasi hasil DBSCAN
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['jumlah_transaksi'], df_cleaned['total_penjualan'],
            c=df_cleaned['dbscan_cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hasil Clustering dengan DBSCAN')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Menampilkan jumlah cluster yang ditemukan oleh DBSCAN
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
st.write(f'Jumlah cluster yang ditemukan: {num_clusters}')
st.write(f'Jumlah noise (outliers): {list(dbscan_labels).count(-1)}')

"""# **Agglomerative Clustering**"""


# Inisialisasi AgglomerativeClustering
agg_clust = AgglomerativeClustering(n_clusters=3)  # Tentukan jumlah cluster

# Pastikan scaled_data memiliki jumlah baris yang sama dengan df
scaled_data = scaler.fit_transform(df[['jumlah_transaksi', 'total_penjualan']])

# Melakukan clustering pada data yang sudah di-scale
agg_labels = agg_clust.fit_predict(scaled_data)

# Menambahkan hasil clustering Agglomerative ke dalam DataFrame
df['agg_cluster'] = agg_labels

# Visualisasi hasil Agglomerative Clustering
plt.scatter(df['jumlah_transaksi'], df['total_penjualan'],
            c=df['agg_cluster'], cmap='plasma')
plt.title('Hasil Clustering dengan Agglomerative')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Menampilkan rata-rata dari masing-masing cluster untuk evaluasi
agg_cluster_summary = df.groupby('agg_cluster')[
    ['jumlah_transaksi', 'total_penjualan']].mean()
st.write(agg_cluster_summary)

"""# **Membandingkan Hasil Clustering KMeans, DBSCAN, dan Agglomerative**

Menghitung Statistik Deskriptif Setiap Cluster
"""

# Menghitung statistik deskriptif untuk masing-masing cluster
cluster_summary = df.groupby('agg_cluster')[
    ['jumlah_transaksi', 'total_penjualan']].describe()
st.write(cluster_summary)

# Memeriksa dan menangani nilai yang hilang
df_cleaned = df.dropna()

# Menggunakan StandardScaler untuk melakukan normalisasi data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    df_cleaned[['jumlah_transaksi', 'total_penjualan']])

"""Silhouette Score"""


# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['kmeans_cluster'] = kmeans.fit_predict(scaled_data)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_cleaned['dbscan_cluster'] = dbscan.fit_predict(scaled_data)

# Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3)
df_cleaned['agg_cluster'] = agg_clust.fit_predict(scaled_data)

# Menghitung Silhouette Score untuk KMeans
kmeans_silhouette = silhouette_score(scaled_data, df_cleaned['kmeans_cluster'])
st.write(f'Silhouette Score untuk KMeans: {kmeans_silhouette:.3f}')

# Menghitung Silhouette Score untuk DBSCAN
# DBSCAN dapat menghasilkan label -1 untuk noise, jadi kita perlu memisahkan noise dari perhitungan
# Pastikan ada lebih dari satu cluster
if len(set(df_cleaned['dbscan_cluster'])) > 1:
    dbscan_silhouette = silhouette_score(
        scaled_data, df_cleaned['dbscan_cluster'])
else:
    # Jika hanya ada satu cluster (atau noise), set score ke -1
    dbscan_silhouette = -1
st.write(f'Silhouette Score untuk DBSCAN: {dbscan_silhouette:.3f}')

# Menghitung Silhouette Score untuk Agglomerative Clustering
agg_silhouette = silhouette_score(scaled_data, df_cleaned['agg_cluster'])
st.write(f'Silhouette Score untuk Agglomerative Clustering: {agg_silhouette:.3f}')

"""Davies-Bouldin Index"""


# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['kmeans_cluster'] = kmeans.fit_predict(scaled_data)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_cleaned['dbscan_cluster'] = dbscan.fit_predict(scaled_data)

# Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=3)
df_cleaned['agg_cluster'] = agg_clust.fit_predict(scaled_data)

# Menghitung Davies-Bouldin Index untuk KMeans
kmeans_dbi = davies_bouldin_score(scaled_data, df_cleaned['kmeans_cluster'])
st.write(f'Davies-Bouldin Index untuk KMeans: {kmeans_dbi:.3f}')

# Menghitung Davies-Bouldin Index untuk DBSCAN
# DBSCAN dapat menghasilkan label -1 untuk noise, jadi kita perlu memisahkan noise dari perhitungan
# Pastikan ada lebih dari satu cluster
if len(set(df_cleaned['dbscan_cluster'])) > 1:
    dbscan_dbi = davies_bouldin_score(
        scaled_data, df_cleaned['dbscan_cluster'])
else:
    # Jika hanya ada satu cluster (atau noise), set score ke infinity
    dbscan_dbi = float('inf')
st.write(f'Davies-Bouldin Index untuk DBSCAN: {dbscan_dbi:.3f}')

# Menghitung Davies-Bouldin Index untuk Agglomerative Clustering
agg_dbi = davies_bouldin_score(scaled_data, df_cleaned['agg_cluster'])
st.write(f'Davies-Bouldin Index untuk Agglomerative Clustering: {agg_dbi:.3f}')

"""Inertia (SSE) - Khusus untuk KMeans"""

# Menghitung Inertia (SSE) untuk KMeans
kmeans_inertia = km.inertia_
st.write(f'Inertia (SSE) untuk KMeans: {kmeans_inertia}')

"""Visualisasi dan Analisis Hubungan Antar Cluster"""


# Visualisasi distribusi data pada setiap cluster
sns.pairplot(df, hue='agg_cluster', vars=[
             'jumlah_transaksi', 'total_penjualan'])
st.pyplot(plt)

"""Jumlah Cluster yang Ditemukan"""

# KMeans
# Menghitung jumlah cluster KMeans
kmeans_clusters = len(set(df_cleaned['kmeans_cluster']))
st.write(f'Jumlah cluster yang ditemukan oleh KMeans: {kmeans_clusters}')

# DBSCAN
# Menghitung jumlah cluster DBSCAN, mengabaikan noise
dbscan_clusters = len(set(df_cleaned['dbscan_cluster'])) - \
    (1 if -1 in df_cleaned['dbscan_cluster'] else 0)
st.write(f'Jumlah cluster yang ditemukan oleh DBSCAN: {dbscan_clusters}')

# Agglomerative Clustering
# Menghitung jumlah cluster Agglomerative
agg_clusters = len(set(df_cleaned['agg_cluster']))
st.write(
    f'Jumlah cluster yang ditemukan oleh Agglomerative Clustering: {agg_clusters}')

"""Visualisasi Hasil Clustering"""


# Visualisasi KMeans
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['jumlah_transaksi'], df_cleaned['total_penjualan'],
            c=df_cleaned['kmeans_cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hasil Clustering dengan KMeans')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
plt.grid(True)
st.pyplot(plt)

# Visualisasi DBSCAN
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['jumlah_transaksi'], df_cleaned['total_penjualan'],
            c=df_cleaned['dbscan_cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hasil Clustering dengan DBSCAN')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
plt.grid(True)
st.pyplot(plt)

# Visualisasi Agglomerative Clustering
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['jumlah_transaksi'], df_cleaned['total_penjualan'],
            c=df_cleaned['agg_cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Hasil Clustering dengan Agglomerative')
plt.xlabel('Jumlah Transaksi')
plt.ylabel('Total Penjualan')
plt.colorbar(label='Cluster')
plt.grid(True)
st.pyplot(plt)
