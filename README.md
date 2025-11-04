# Proyek Detektor Wajah: LBP + SVM dengan Overlay Topi Virtual

Ini adalah proyek computer vision klasik yang mengimplementasikan deteksi wajah real-time dan aplikasi virtual try-on (uji coba virtual).

Proyek ini menggunakan backend Python (OpenCV, Scikit-learn) untuk melakukan pemrosesan video dan deteksi wajah. Stream video yang telah diproses kemudian dikirim melalui jaringan (UDP) ke frontend Godot Engine, di mana pengguna dapat secara interaktif memilih berbagai topi untuk "dicoba".

## Arsitektur Pipeline

Proyek ini menggunakan pipeline deteksi hibrid untuk mencapai keseimbangan antara kecepatan dan akurasi.

### Tahap 1: Proposal (Haar Cascade)

Setiap frame dari webcam pertama kali dianalisis menggunakan classifier Haar Cascade (`haarcascade_frontalface_default.xml`) dari OpenCV.

Metode ini sangat cepat dan efisien dalam membuang area yang pasti bukan wajah, serta memberikan beberapa "kandidat" ROI (Region of Interest) yang mungkin wajah.

### Tahap 2: Verifikasi (LBP + SVM)

Setiap ROI kandidat dari Tahap 1 kemudian dianalisis oleh classifier kita yang telah dilatih.

- **LBP (Local Binary Patterns)**: Fitur tekstur diekstraksi dari ROI. LBP sangat baik dalam mendeskripsikan tekstur wajah dan tahan terhadap perubahan pencahayaan.
- **SVM (Support Vector Machine)**: Vektor fitur LBP kemudian dimasukkan ke model SVM (`svm_lbp.pkl`) yang telah kita latih untuk membuat keputusan akhir: "Ya, ini wajah" (Label 1) atau "Bukan, ini false positive" (Label 0).

Arsitektur "Proposal + Verifikasi" ini jauh lebih cepat daripada sliding window murni, sehingga memungkinkan deteksi real-time.

## 📁 Struktur Folder Proyek
```
svm_lbp_hat/
├── assets/
│   ├── cascades/
│   │   ├── haarcascade_frontalface_default.xml
│   │   └── haarcascade_eye.xml
│   └── hats/
│       ├── top_hat.png
│       ├── top_hat.json
│       ├── sombrero.png
│       ├── sombrero.json
│       └── (dan topi lainnya...)
│
├── data/
│   ├── faces/              # Data latih wajah
│   └── non_faces/          # Data latih non-wajah
│
├── models/
│   ├── svm_lbp.pkl         # Dihasilkan oleh 'app.py train'
│   └── test_data.pkl       # Dihasilkan oleh 'app.py train'
│
├── pipelines/
│   ├── __init__.py
│   ├── dataset.py          # Memuat data latih
│   ├── features.py         # Logika ekstraksi LBP
│   ├── infer.py            # Logika pipeline deteksi
│   ├── overlay.py          # Logika penempatan topi
│   ├── train.py            # Logika training SVM/RF
│   └── utils.py            # Fungsi helper, pemuat aset
│
├── app.py                  # Entry point untuk training & tes lokal
├── run_server.py           # Entry point untuk Server UDP
├── client.py               # Client Python sederhana untuk debugging
├── webcam_client_udp.gd    # Skrip untuk Godot Client
└── requirements.txt        # Dependensi Python
```

## 🚀 Setup Proyek

### Bagian 1: Backend (Python Server)

1. **Klon/Buat Proyek**: Siapkan struktur folder di atas.

2. **Buat Virtual Environment**:
```bash
   python -m venv venv
   venv\Scripts\activate  # (Windows)
   # source venv/bin/activate # (Mac/Linux)
```

3. **Instal Dependensi**:
```bash
   pip install -r requirements.txt
```

4. **Unduh Aset Cascade**:
   - Unduh `haarcascade_frontalface_default.xml` dan `haarcascade_eye.xml` dari [GitHub OpenCV](https://github.com/opencv/opencv/tree/master/data/haarcascades) atau dari library yang telah diinstal.
   - Tempatkan kedua file tersebut di `assets/cascades/`.

### Bagian 2: Aset Topi (PENTING!)

Aplikasi ini menggunakan sistem metadata JSON untuk mengatur posisi dan ukuran setiap topi.

1. Tempatkan semua gambar topi Anda di `assets/hats/`.
2. Untuk setiap file gambar (misal `top_hat.png`), buat file `.json` dengan nama yang sama persis (misal `top_hat.json`).
3. Isi file `.json` dengan parameter penyesuaian.

**Contoh `assets/hats/top_hat.json`**:
```json
{
  "scale_factor": 1.4,
  "y_offset_factor": 0.8,
  "x_offset_factor": 0.0
}
```

- **`scale_factor`**: Ukuran topi. 1.0 = 100% lebar kotak wajah.
- **`y_offset_factor`**: Posisi vertikal. Nilai lebih besar = lebih tinggi (mengambang). Nilai lebih kecil = lebih rendah.
- **`x_offset_factor`**: Posisi horizontal. 0.0 = tengah. Nilai positif = geser ke kanan. Nilai negatif = geser ke kiri.

**PENTING**: Nama file Anda akan otomatis diubah menjadi nama Kategori.
- `top_hat.png` akan menjadi "TOP HAT".
- `pith_helmet.png` akan menjadi "PITH HELMET".
- `zucchetto.png` akan menjadi "ZUCCHETTO".

Pastikan nama file ini cocok dengan teks pada tombol-tombol di Godot Anda.

### Bagian 3: Frontend (Godot Client)

1. Buka proyek Godot Anda.
2. Lampirkan skrip `webcam_client_udp.gd` ke node utama scene UI Anda.
3. Pastikan nama-nama node di skrip (misal `@onready var texture_rect = ...`) cocok dengan nama node di scene tree Anda.

## 💻 Cara Menjalankan Proyek

### Langkah 1: Latih Model Anda (Hanya 1x)

Anda harus memiliki data di `data/faces` dan `data/non_faces`.
```bash
# Buka terminal dan jalankan:
python app.py train --pos_dir data/faces --neg_dir data/non_faces
```

Ini akan membuat file `svm_lbp.pkl` (atau `rf_lbp.pkl` jika Anda menggunakan `--classifier rf`) di dalam folder `models/`.

### Langkah 2: Jalankan Server (Python)

Pastikan `svm_lbp.pkl` Anda sudah ada. Server ini akan memuat model, menyalakan webcam, dan mulai men-streaming video.
```bash
# Di terminal yang sama, jalankan:
python run_server.py
```

Anda akan melihat log: `🚀 UDP Server started at 0.0.0.0:8888`.

### Langkah 3: Jalankan Client (Godot)

1. Buka proyek Godot Anda.
2. Jalankan scene utama (F5).
3. Klik tombol "Connect to Server".
4. Anda akan melihat stream video dari server Python Anda.
5. Klik tombol kategori topi (misal "TOP HAT") untuk menampilkan topi.
6. Klik tombol yang sama lagi untuk menyembunyikan topi (HAT_OFF).

