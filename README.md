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
Penelitian ini bertujuan untuk mengaplikasikan teknik Data Mining, khususnya algoritma K-Means Clustering, dalam menganalisis pola transaksi penjualan di minimarket. Dataset publik dari Kaggle digunakan dengan atribut utama seperti kode barang, nama barang, jumlah transaksi, total penjualan, dan rata-rata penjualan. Tujuan utama dari penelitian ini adalah untuk mengidentifikasi segmen pelanggan, mengoptimalkan strategi pemasaran, serta membantu pengelolaan stok barang yang lebih efisien. Hasil analisis divisualisasikan untuk mendukung pengambilan keputusan berbasis data.

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
Dataset yang digunakan dalam penelitian ini adalah dataset barang_keluar yang diambil dari Kaggle. Dataset terdiri dari informasi mengenai transaksi barang yang dijual di minimarket, dengan atribut-atribut utama sebagai berikut:
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
1. **Langkah-langkah**
   - Inisialisasi centroid awal secara acak.
   - Iterasi untuk meminimalkan jarak data terhadap centroid hingga mencapai konvergensi.
2. **Evaluasi Model**
   - Metode **Elbow** digunakan untuk menentukan jumlah klaster optimal dengan melihat perubahan inertia.
   - Mengevaluasi hasil distribusi klaster dengan visualisasi scatter plot.

3. **Hasil dan Visualisasi**
   - Scatter plot klaster untuk menampilkan distribusi data.
   - Analisis karakteristik klaster berdasarkan rata-rata jumlah transaksi dan total penjualan pada tiap klaster.

---

## ** Performa Model**
Hasil dari algoritma K-Means Clustering menunjukkan bahwa dataset terbagi menjadi 3 klaster utama berdasarkan pola pembelian, yaitu:
- Klaster 1: Pelanggan dengan transaksi rendah.
- Klaster 2: Pelanggan dengan transaksi menengah.
- Klaster 3: Pelanggan dengan transaksi tinggi.
Model ini menunjukkan hasil yang baik dengan error yang minim, dan evaluasi menggunakan Elbow Method menghasilkan jumlah klaster optimal sebanyak 3.

---

## **Hasil dan Kesimpulan**
1. Dataset terbagi menjadi 3 klaster utama berdasarkan jumlah transaksi dan total penjualan:
   - **Klaster 1**: Pelanggan dengan transaksi rendah.
   - **Klaster 2**: Pelanggan dengan transaksi menengah.
   - **Klaster 3**: Pelanggan dengan transaksi tinggi.

2. **Rekomendasi Strategi Pemasaran**:
   - **Klaster 1**: Berikan promosi diskon untuk menarik minat pelanggan.
   - **Klaster 2**: Berikan penawaran loyalitas, seperti voucher diskon untuk pembelian berikutnya.
   - **Klaster 3**: Tawarkan produk eksklusif atau layanan tambahan untuk meningkatkan pengalaman pelanggan.

---

## **Kesimpulan Akhir**
Penerapan algoritma **K-Means Clustering** pada dataset transaksi penjualan memberikan wawasan mendalam mengenai pola pembelian pelanggan. Hasil klasterisasi membantu memformulasikan strategi pemasaran yang lebih efektif untuk tiap segmen pelanggan. Penelitian ini menunjukkan bahwa analisis data berbasis klaster dapat mendukung pengambilan keputusan yang lebih baik dalam pengelolaan stok barang dan pemasaran.

