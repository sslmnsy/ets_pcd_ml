# Virtual Try-On Topi (LBP+SVM) dengan Godot

Proyek ini adalah implementasi *virtual try-on* topi (VTO) secara *real-time* yang menggunakan arsitektur *client-server*.

* **Backend (Server):** Ditulis dalam **Python** menggunakan OpenCV dan Scikit-learn. Bertugas menjalankan *pipeline* deteksi wajah yang intensif secara komputasi.
* **Frontend (Client):** Dibangun dengan **Godot Engine**. Bertugas sebagai antarmuka pengguna (UI) yang interaktif dan menampilkan hasil *streaming* video.

Proyek ini menggunakan metode *Computer Vision* klasik (Haar Cascades, LBP, dan SVM) dan **tidak** menggunakan *Deep Learning*.

## 🖼️ Tampilan Aplikasi

| Menu Utama (`main_menu.tscn`) | Try-On Topi (`webcam_ui.tscn`) |
| :---: | :---: |
| <img width="2780" height="1555" alt="image" src="https://github.com/user-attachments/assets/7098346b-d002-480e-8c16-42fdb0ef002e" /> | <img width="2767" height="1557" alt="image" src="https://github.com/user-attachments/assets/4832871f-0156-45ba-bcfa-df259cb45d32" />
|

## 🚀 Arsitektur & Alur Kerja

Sistem ini menggunakan arsitektur *client-server* untuk memisahkan logika UI (ringan) dari logika CV (berat). Komunikasi berlangsung melalui *streaming* video UDP.

<img width="896" height="723" alt="image" src="https://github.com/user-attachments/assets/209e0e38-1847-469f-af20-b0445700404f" />

**Alur Logika (Pipeline):**

1. **Server (Python):** `run_server.py` mengakses webcam (`cv2.VideoCapture`).
2. **Tahap 1: Proposal ROI (Haar Cascade):** *Server* menggunakan Haar Cascade (`haarcascade_frontalface_default.xml`) untuk menemukan *kandidat* wajah dengan cepat.
3. **Tahap 2: Verifikasi Fitur (LBP+SVM):**
   * Setiap kandidat ROI dianalisis fiturnya menggunakan **Local Binary Patterns (LBP)**.
   * Vektor fitur LBP dikirim ke *classifier* **SVM** (`svm_lbp.pkl`) yang telah dilatih.
4. **Overlay (Python):** Jika SVM memverifikasi wajah, *server* mengambil aset topi (`.png`) dan `.json`-nya (untuk penempatan) lalu menggambarnya di atas *frame* video.
5. **Streaming (Python):** *Frame* yang sudah jadi (video + topi) di-*encode* ke **JPEG** dan di-*stream* melalui UDP.
6. **Tampilan (Godot):** *Client* Godot menerima paket-paket JPEG, merakitnya kembali, dan menampilkannya di `TextureRect`.
7. **Interaksi (Godot):** Pengguna menekan tombol topi di UI Godot. Godot mengirimkan **perintah teks** (misal, `"HAT_CATEGORY:TOP HAT"`) ke *server* Python, yang kemudian mengganti topi di langkah #4.

## 📁 Struktur Folder Proyek
```
ets_pcd_ml/
│
├── 📂 assets/
│   ├── 📂 cascades/
│   │   ├── haarcascade_frontalface_default.xml   (Detektor Wajah Cepat)
│   │   └── haarcascade_eye.xml                   (Untuk Rotasi Topi)
│   └── 📂 hats/
│       ├── top_hat.png                           (Gambar Aset)
│       ├── top_hat.json                          (Metadata Posisi)
│       ├── sombrero.png
│       └── sombrero.json                         (dst...)
│
├── 📂 data/
│   ├── 📂 faces/                                 (Data latih positif - hasil crop)
│   └── 📂 non_faces/                             (Data latih negatif)
│
├── 📂 godot_client/
│   ├── main_menu.tscn
│   ├── webcam_ui.tscn
│   ├── guide.tscn
│   ├── about_team.tscn
│   ├── webcam_client_udp.gd                      (Skrip utama Godot)
│   └── project.godot
│
├── 📂 models/
│   ├── svm_lbp.pkl                               (Model SVM yang dilatih)
│   └── test_data.pkl                             (Data uji untuk evaluasi)
│
├── 📂 pipelines/
│   ├── infer.py                                  (Logika pipeline deteksi)
│   ├── train.py                                  (Logika pipeline training)
│   ├── features.py                               (Logika ekstraksi LBP)
│   ├── dataset.py                                (Logika memuat data)
│   ├── overlay.py                                (Logika menempelkan topi)
│   └── utils.py                                  (Fungsi helper)
│
├── app.py                                        (Utilitas: train, eval, webcam lokal)
├── run_server.py                                 (Aplikasi Server Utama - untuk Godot)
├── client.py                                     (Client Python sederhana untuk debug)
├── preprocess.py                                 (Skrip untuk cropping/cleaning dataset)
└── requirements.txt                              (Dependensi Python)
```

## ⚙️ Instalasi dan Setup

### 1. Backend (Python)

Disarankan untuk menggunakan *virtual environment* (venv).
```bash
# 1. Arahkan ke folder proyek
cd ets_pcd_ml

# 2. Buat venv baru
python -m venv venv

# 3. Aktifkan venv
.\venv\Scripts\activate

# 4. Instal semua library yang dibutuhkan
pip install -r requirements.txt
```

### 2. Frontend (Godot)

1. Buka Godot Engine (v4.x).
2. Klik "Import" atau "Scan".
3. Arahkan ke folder `godot_client/` dan impor file `project.godot`.

## 🚀 Cara Menjalankan Proyek (Alur Kerja Lengkap)

Proyek ini harus dijalankan dalam 4 tahap:

### Tahap 1: Persiapan Data (Hanya sekali)

1. Gunakan folder `data/faces` dan `data/non_faces` yang sudah tersedia, atau jika ingin menggunakan data sendiri,
2. Unduh dataset gambar mentah (misal Caltech 10k untuk wajah, ImageNet/Kaggle untuk non-wajah).
3. Jalankan skrip `preprocess.py` (yang berisi `DatasetPreprocessor` dan `DatasetCleaner`).
```bash
   python preprocess.py
```
3. Ikuti petunjuk di terminal untuk melakukan *cropping* wajah dari data mentah dan *cleaning* (membatasi jumlah) data latih Anda.
### Tahap 2: Training Model (Hanya sekali)

Setelah `data/faces` dan `data/non_faces` Anda siap, latih model SVM Anda.
```bash
# Jalankan 'app.py' dengan perintah 'train'
python app.py train --pos_dir data/faces --neg_dir data/non_faces
```

Proses ini akan memakan waktu beberapa menit dan menghasilkan `models/svm_lbp.pkl` dan `models/test_data.pkl`.

### Tahap 3: Menjalankan Server Backend

Sekarang Anda bisa menjalankan server utama. Server ini akan mengakses webcam Anda.
```bash
python run_server.py
```

Terminal akan menampilkan `🚀 UDP Server started at 0.0.0.0:8888`. Server sekarang sedang *streaming*.

### Tahap 4: Menjalankan Client Frontend

1. Buka proyek `godot_client/` di Godot Engine.
2. Jalankan *scene* utama (`main_menu.tscn`).
3. Tekan tombol **"Mulai Try-On"**.
4. Di *scene* webcam, tekan tombol **"Connect to Server"**.
5. Video Anda akan muncul, dan Anda dapat mulai memilih topi.

## 🎩 Cara Menambahkan Topi Baru

Sistem ini sepenuhnya dinamis. Untuk menambahkan topi baru:

1. **Tambahkan Gambar:** Simpan gambar topi transparan Anda di `assets/hats/`. Contoh: `cowboy_hat.png`.

2. **Tambahkan JSON:** Buat file `.json` dengan **nama yang sama persis** di folder yang sama. Contoh: `cowboy_hat.json`.

3. **Isi JSON:** Edit file `.json` dengan parameter penempatan.

   **Contoh `cowboy_hat.json`:**
```json
   {
     "scale_factor": 1.5,
     "y_offset_factor": 0.85,
     "x_offset_factor": -0.05
   }
```

   * `scale_factor`: Mengontrol **ukuran**. > 1.0 = lebih besar.
   * `y_offset_factor`: Mengontrol **posisi vertikal**. Nilai lebih besar = lebih rendah.
   * `x_offset_factor`: Mengontrol **posisi horizontal**. Nilai positif = ke kanan.

4. **Selesai.** Anda hanya perlu me-restart `run_server.py` agar ia memuat topi baru Anda. Anda **tidak perlu** mengedit kode Godot (UI Anda akan butuh tombol baru, tapi server akan langsung mengenali file `cowboy_hat.png` jika nama filenya diubah menjadi `COWBOY HAT`).


## 👥 Tim Pengembang
Fitri Salwa
Salma Nesya Putri Salia
