**DOKUMENTASI PROJECT DATA MINING**

# **Analisis Klasterisasi pada Transaksi Penjualan**
**Mengidentifikasi Pola Pembelian Menggunakan Algoritma K-Means Clustering**

### **Disusun untuk Memenuhi Tugas Mata Kuliah Data Mining**

**Dosen Pengampu:**  
Abu Salam, M.Kom

**Disusun Oleh:**  
Naufal Arsyaputra Pradana  
A11.2022.14606

**Program Studi Teknik Informatika**  
Fakultas Ilmu Komputer  
Universitas Dian Nuswantoro  
2024/2025

---

## **Ringkasan**
Penelitian ini bertujuan untuk mengaplikasikan teknik Data Mining, dengan fokus pada algoritma K-Means Clustering (sebagai model dari Unsupervised Learning), untuk menganalisis pola transaksi penjualan di sebuah minimarket. Data yang digunakan berupa dataset publik dari Kaggle, yang memuat informasi pembelian pelanggan, meliputi lima atribut utama: kode barang, nama barang, jumlah transaksi, total penjualan, dan rata-rata penjualan. Dataset ini berbentuk data tabular dengan mayoritas nilai kuantitatif.
Melalui penerapan algoritma K-Means Clustering, penelitian ini diharapkan mampu mengidentifikasi segmen pelanggan berdasarkan pola belanja mereka, memberikan wawasan yang mendalam untuk mengoptimalkan strategi pemasaran, serta membantu pengelolaan stok barang yang lebih efisien. Hasil analisis akan divisualisasikan secara informatif, sehingga memudahkan pengambilan keputusan yang berbasis data.

---

## **Permasalahan**
Data transaksi sering kali tidak terorganisir, sehingga sulit untuk dianalisis. Hal ini menghambat pengelolaan stok dan strategi pemasaran yang optimal. Penelitian ini mencoba memberikan solusi melalui klasterisasi pelanggan berdasarkan pola pembelian. Beberapa permasalahan yang diidentifikasi antara lain:
1. Sulitnya mengidentifikasi barang yang kurang laku atau barang dengan permintaan tinggi.
2. Tidak adanya segmentasi pelanggan berdasarkan pola belanja.
3. Strategi pemasaran yang kurang terarah akibat tidak adanya analisis data yang mendalam.

---

## **Tujuan**
Penelitian ini memiliki beberapa tujuan utama:
1. Mengimplementasikan algoritma **K-Means Clustering** untuk klasterisasi data transaksi.
2. Mengidentifikasi pola pembelian pelanggan berdasarkan klaster yang terbentuk berdasarkan hasil klasterisasi.
3. Menyajikan visualisasi hasil klasterisasi yang mudah dipahami dan berguna untuk pengambilan keputusan.
4. Memberikan rekomendasi strategi pemasaran berdasarkan hasil analisis klasterisasi.

---

## **Model / Alur Penyelesaian**
1. **Pengumpulan Data**: Data transaksi diambil dari Kaggle.
2. **Pembersihan Data**: Menghapus duplikasi dan nilai kosong.
3. **Pra-Proses Data**: Melakukan normalisasi dan seleksi fitur untuk meningkatkan kualitas data.
4. **Penerapan K-Means Clustering**: Klasterisasi data dengan K-Means.
5. **Evaluasi Model**: Menggunakan metode Elbow untuk menentukan jumlah klaster optimal.
6. **Visualisasi Hasil**: Menampilkan hasil klasterisasi menggunakan scatter plot dan diagram lainnya.
7. **Rekomendasi Strategi Pemasaran**: Mengusulkan strategi pemasaran berbasis hasil klasterisasi.

flowchart:

    Start([Mulai]) --> Data[Pengumpulan Data]
    Data --> Preprocess[Pra-Proses Data]
    Preprocess -->|Cleansing, Selection, Transform| Modeling[Penerapan Algoritma K-Means]
    Modeling --> Evaluation[Evaluasi dan Analisis Hasil]
    Evaluation --> Visualization[Visualisasi Data]
    Visualization --> Recommendation[Rekomendasi dan Keputusan]
    Recommendation --> End([Selesai])

## **Tahapan Penelitian**
1. **Pengumpulan Data**
   - Dataset diambil dari Kaggle.
   - Atribut penting: jumlah transaksi, total penjualan, rata-rata penjualan.

2. **Pra-Proses Data**
   - Membersihkan data (menghapus duplikasi dan mengisi nilai kosong).
   - Menormalkan data menggunakan teknik seperti MinMax Scaling.
   - Seleksi atribut yang relevan untuk meningkatkan akurasi model.

3. **Penerapan K-Means Clustering**
   - Melakukan klasterisasi berdasarkan jumlah transaksi dan total penjualan.
   - Mengevaluasi hasil klasterisasi menggunakan metode Elbow untuk menentukan jumlah klaster optimal.

4. **Visualisasi Hasil**
   - Scatter plot untuk distribusi klaster.
   - Diagram batang/pie chart untuk analisis segmen pelanggan.

5. **Rekomendasi Strategi**
   - Mengusulkan strategi pemasaran berdasarkan hasil klasterisasi, seperti promosi spesifik per segmen pelanggan.

---

## **Penjelasan Dataset**
Dataset yang digunakan dalam eksperimen ini merupakan dataset transaksi penjualan yang berasal dari platform Kaggle dengan nama barang_keluar. Dataset ini mencakup sekitar 7.400 baris data yang menggambarkan transaksi penjualan barang oleh pelanggan dalam suatu periode tertentu. Data ini masih dalam bentuk raw data yang memerlukan beberapa tahap pembersihan dan transformasi agar dapat digunakan secara efektif dalam analisis lebih lanjut.
Berikut adalah struktur dan penjelasan dari masing-masing kolom dalam dataset:
- Dataset: **barang_keluar** dari Kaggle.
- Total data: 7.400 baris.
- Struktur data:
  - **kode_barang**: Kode unik untuk setiap barang.
  - **nama_barang**: Nama barang yang dijual.
  - **jumlah_transaksi**: Frekuensi penjualan per barang.
  - **total_penjualan**: Total unit yang terjual.
  - **rata_rata**: Rata-rata unit per transaksi.

---

## **Exploratory Data Analysis (EDA)**
1. **Memeriksa Tipe Data dan Statistik Deskriptif**
   - Menghitung rata-rata, median, standar deviasi, dan kuartil untuk data numerik.
2. **Visualisasi Data**
   - Histogram: Distribusi data jumlah transaksi dan total penjualan.
   - Box plot: Deteksi outlier untuk setiap atribut numerik.
   - Scatter plot: Hubungan antar fitur untuk melihat pola klaster awal.
3. **Korelasi Antar Fitur**
   - Menggunakan koefisien Pearson untuk mengidentifikasi hubungan antar atribut numerik.
4. **Pencarian Missing Data**
   - Memastikan dataset bebas dari nilai kosong atau outlier yang signifikan.

---

## **Proses Features Dataset**
1. **Pembersihan Data**
   - Menghapus duplikasi dan nilai kosong untuk memastikan data bersih.
2. **Transformasi Data**
   - Normalisasi fitur menggunakan **MinMax Scaling** untuk menyetarakan skala data.
   - Encoding kategori (jika ada) untuk fitur non-numerik.
3. **Seleksi Fitur**
   - Menggunakan korelasi untuk memilih atribut paling relevan dengan tujuan penelitian.
4. **Pembagian Dataset**
   - Data latih (80%) dan data uji (20%) untuk validasi model.

---

## **Proses Modelling dengan K-Means Clustering**
1. **Pemuatan dan Pemeriksaan Data**
   - Pembacaan Data: Data dimuat dari file CSV menggunakan pandas. File tersebut berisi informasi terkait transaksi penjualan barang. Pemisah yang digunakan dalam CSV adalah titik koma (;), dan low_memory=False memastikan bahwa dataset besar dapat diproses dengan benar tanpa masalah pemrosesan memori.
   - Pemeriksaan Data: Setelah data dimuat, kita memeriksa lima baris pertama menggunakan df.head() dan melihat informasi umum tentang dataset dengan df.info(). Ini bertujuan untuk memeriksa apakah data telah dimuat dengan benar dan memahami struktur serta tipe data yang ada.
2. **Pembersihan Data**
   - Menghapus Nilai Kosong: Pada tahap ini, kita memeriksa apakah ada nilai yang hilang atau NaN dalam dataset dengan df.isnull().sum(). Jika ditemukan, kita menghapus baris yang mengandung nilai kosong menggunakan dropna(). Hal ini penting untuk mencegah gangguan dalam pemodelan dan analisis data.
   - Menghapus Duplikat: Kita juga memeriksa dan menghapus baris duplikat untuk memastikan bahwa data yang digunakan dalam pemodelan adalah unik dan tidak terdistorsi oleh pengulangan.
3. **Seleksi Fitur**
   - Seleksi Fitur: Dari seluruh dataset, kita hanya memilih dua fitur utama, yaitu jumlah_transaksi dan total_penjualan, yang akan digunakan untuk analisis clustering. Pilihan ini dibuat dengan alasan bahwa kedua variabel tersebut terkait langsung dengan tujuan pemodelan yaitu mengelompokkan transaksi berdasarkan perilaku penjualan.
4. **Normalisasi Datar**
   - Normalisasi: Proses ini dilakukan untuk memastikan bahwa fitur yang digunakan dalam clustering memiliki skala yang serupa. K-Means mengandalkan jarak Euclidean antar titik, dan fitur dengan rentang yang lebih besar (misalnya jumlah_transaksi) dapat mendominasi jarak antar data. Normalisasi menggunakan MinMaxScaler mengubah rentang setiap fitur agar berada di antara 0 dan 1, menghindari distorsi akibat perbedaan skala.
5. **Visualisasi Data**
   - Visualisasi Data: Sebelum memulai clustering, sangat penting untuk memahami distribusi data. Dengan menggunakan scatter plot, kita bisa memvisualisasikan hubungan antara Jumlah Transaksi dan Total Penjualan. Ini juga membantu kita dalam melihat potensi pola atau kelompok yang dapat terbentuk dalam data.
7. **Menentukan Jumlah Cluster Optimal (Elbow Method)**
   - Metode Elbow: Salah satu tantangan dalam K-Means adalah menentukan jumlah cluster yang optimal. Untuk itu, kita menggunakan Elbow Method, yang melibatkan pengujian beberapa nilai k (jumlah cluster) dan mengukur inertia (jumlah total jarak antara titik data dan centroid mereka). Semakin kecil inertia, semakin baik clustering yang dihasilkan. Namun, setelah suatu titik tertentu, penurunan inertia cenderung melambat, yang membentuk bentuk siku atau "elbow". Titik ini menjadi indikasi jumlah cluster yang optimal
8. **Menerapkan K-Means Clustering**
   - Melakukan Clustering: Setelah menentukan jumlah cluster yang optimal (misalnya 3 cluster), kita menjalankan algoritma K-Means dengan parameter n_clusters=3. Metode fit_predict() digunakan untuk melatih model dan memprediksi cluster yang akan dihasilkan untuk setiap data.
9. **Visualisasi Hasil Clustering**
    - Visualisasi Clustering: Setelah proses clustering, kita visualisasikan hasilnya menggunakan scatter plot, di mana setiap titik data diberi warna yang berbeda sesuai dengan cluster-nya. Ini memberi kita gambaran tentang bagaimana data dikelompokkan dan pola apa yang ditemukan.
10. **Evaluasi Model: Silhouette Score**
    - Silhouette Score: Setelah melakukan clustering, kita mengevaluasi kualitas clustering menggunakan Silhouette Score. Nilai ini mengukur seberapa baik suatu objek diklasifikasikan ke dalam cluster yang benar. Nilai Silhouette berkisar antara -1 hingga 1, di mana nilai yang lebih tinggi menunjukkan bahwa clustering yang dilakukan lebih baik.

---

## ** Performa Model**
Evaluasi performa model K-Means Clustering penting untuk memastikan bahwa proses clustering memberikan hasil yang baik dan sesuai dengan tujuan analisis.
Berikut ini adalah penjelasan lengkap tentang evaluasi performa model berdasarkan beberapa metrik dan analisis yang dilakukan.:
- Evaluasi dengan Silhouette Score: Silhouette Score mengukur seberapa baik setiap titik data berada dalam cluster yang benar, relatif terhadap cluster lain. Skor ini berkisar dari -1 hingga 1
- Visualisasi Cluster : PSetelah clustering selesai, hasilnya divisualisasikan menggunakan scatter plot dengan warna berbeda untuk setiap cluster.
- Evaluasi Dengan Inertia: Inertia adalah total jarak kuadrat antara setiap titik data dan centroid cluster-nya. Semakin kecil nilai inertia, semakin baik clustering, tetapi nilai ini hanya berguna untuk membandingkan hasil dari jumlah cluster yang berbeda (misalnya, selama Elbow Method).
- Silhouette Score memberikan indikasi bahwa cluster sudah terbentuk dengan baik (contoh: 0.65 menunjukkan clustering yang baik).
- Visualisasi Cluster menunjukkan distribusi dan jarak antar cluster, di mana cluster terpisah dengan jelas, tetapi mungkin ada beberapa outlier.
- Inertia menunjukkan bahwa model berhasil meminimalkan jarak antara titik data dengan centroid.

---

## **Hasil dan Kesimpulan**
Penelitian ini bertujuan untuk mengelompokkan data transaksi penjualan berdasarkan pola jumlah transaksi dan total penjualan,
1. Dataset terbagi menjadi 3 klaster utama berdasarkan jumlah transaksi dan total penjualan:
   - **Klaster 1**: Pelanggan dengan transaksi rendah.
   - **Klaster 2**: Pelanggan dengan transaksi menengah.
   - **Klaster 3**: Pelanggan dengan transaksi tinggi.

2. **Rekomendasi Strategi Pemasaran**:
   - **Klaster 1**: Berikan promosi diskon untuk menarik minat pelanggan.
   - **Klaster 2**: Berikan penawaran loyalitas, seperti voucher diskon untuk pembelian berikutnya.
   - **Klaster 3**: Tawarkan produk eksklusif atau layanan tambahan untuk meningkatkan pengalaman pelanggan.
- Model ini bergantung pada asumsi bahwa cluster berbentuk bola. Jika pola data lebih kompleks, pendekatan lain seperti DBSCAN atau Gaussian Mixture Model mungkin lebih efektif.
  
---

## **Kesimpulan Akhir**
Penerapan algoritma **K-Means Clustering** pada dataset transaksi penjualan memberikan wawasan mendalam mengenai pola pembelian pelanggan. Hasil klasterisasi membantu memformulasikan strategi pemasaran yang lebih efektif untuk tiap segmen pelanggan. Penelitian ini menunjukkan bahwa analisis data berbasis klaster dapat mendukung pengambilan keputusan yang lebih baik dalam pengelolaan stok barang dan pemasaran.

