# Klasifikasi Citra Dokumen dengan Ensemble Learning (Stacking)

**Laporan Metodologi dan Pemodelan — Kompetisi `document-image-classification`**

> Dokumen ini menjelaskan pipeline dari tiga notebook: `convnext.ipynb` (model dasar ConvNeXt), `dit-base.ipynb` (model dasar DiT), dan `metamodel.ipynb` (meta-model XGBoost + Optuna). Selain kode, laporan ini memakai dua sumber bukti tambahan: **berkas probabilitas level 0** (`oof_*.npy`, `test_*.npy`) dan **tiga berkas log eksekusi Kaggle** (`convnext.log`, `dit-base__1_.log`, `metamodel.log`). Notebook tidak menyimpan keluaran sel dan `train.csv` tidak dilampirkan, sehingga **akurasi model dasar terhadap label asli belum dapat dihitung**; sel yang membutuhkannya ditandai `[isi]`. Angka dari berkas `.npy` dihitung langsung dari berkas tersebut, dan angka dari log disalin atau dirata-ratakan dari keluaran log. Hal yang tidak dapat dibuktikan ditandai sebagai *hipotesis* atau *keterbatasan*.

---

## Daftar Isi

1. [Ringkasan](#1-ringkasan)
2. [Latar Belakang dan Tujuan](#2-latar-belakang-dan-tujuan)
3. [Data](#3-data)
4. [Arsitektur Stacking](#4-arsitektur-stacking)
5. [Level 0: Model Dasar](#5-level-0-model-dasar)
6. [Level 1: Meta-Model](#6-level-1-meta-model)
7. [Hasil](#7-hasil)
8. [Diskusi: Validitas dan Keterbatasan](#8-diskusi-validitas-dan-keterbatasan)
9. [Rekomendasi Tindak Lanjut](#9-rekomendasi-tindak-lanjut)
10. [Struktur Repositori dan Reproduksibilitas](#10-struktur-repositori-dan-reproduksibilitas)

---

## 1. Ringkasan

Proyek ini membangun sistem klasifikasi citra untuk kompetisi *document-image-classification* menggunakan **stacking (ensemble learning)**. Data latih berisi **32.003 gambar** dan data uji berisi **7.994 gambar** (format `.tif`) dengan **16 kelas** (label 0–15; lihat Bagian 3.2).

| Komponen | Isi |
|----------|-----|
| Model dasar (level 0) | ConvNeXt-Base (`timm`) dan DiT-Base (`microsoft/dit-base`, Hugging Face) |
| Skema pelatihan level 0 | Stratified 5-fold, menghasilkan probabilitas *out-of-fold* (OOF) untuk data latih dan rata-rata 5 fold untuk data uji |
| Fitur meta | Gabungan probabilitas kedua model dasar, **2 model × 16 kelas = 32 fitur** |
| Meta-model (level 1) | XGBoost (`multi:softprob`), hiperparameter dicari dengan Optuna (30 trial, 5-fold CV, metrik akurasi) |
| Keluaran | `submission_optuna.csv` (7.994 baris) |
| Hasil CV meta-model (log) | Empat kandidat meta-model dibandingkan dengan CV 5-fold. XGBoost terbaik: akurasi 0,842, Macro F1 0,842, log loss 0,542 (Bagian 6.6) |
| Temuan dari berkas probabilitas | Kedua model memakai ke-16 kelas secara merata; keduanya sepakat pada 79,4% data OOF, sehingga ada ruang bagi stacking (Bagian 7) |

---

## 2. Latar Belakang dan Tujuan

Klasifikasi citra otomatis mengurangi pekerjaan pemilahan manual pada koleksi gambar berskala puluhan ribu. Satu model tunggal cenderung membuat kesalahan yang khas pada arsitekturnya. Model berbasis konvolusi (ConvNeXt) dan model berbasis *transformer* (DiT) belajar representasi yang berbeda, sehingga kesalahan keduanya tidak sepenuhnya sama. Stacking memanfaatkan perbedaan itu dengan melatih meta-model yang belajar kapan harus mempercayai masing-masing model dasar.

**Tujuan:**

1. Melatih dua model dasar yang beragam arsitekturnya pada gambar dokumen.
2. Menghasilkan prediksi OOF yang bebas kebocoran untuk melatih meta-model.
3. Menggabungkan prediksi tersebut dengan meta-model XGBoost yang disetel dengan Optuna, lalu membuat berkas submission.

---

## 3. Data

### 3.1 Sumber dan skala

| Berkas / folder | Ukuran | Keterangan |
|-----------------|--------|------------|
| `train.csv` | 32.003 baris | Kolom yang dipakai: `filename`, `class_name`, `label` |
| `train/train/<class_name>/*.tif` | 32.003 gambar | Satu subfolder per kelas |
| `test/test/*.tif` | 7.994 gambar | Tanpa label |
| `sample_submission.csv` | 7.994 baris | Kolom `id` dan `label`; menentukan urutan submission |

### 3.2 Kelas

Data memiliki **16 kelas** jenis dokumen (label 0–15), sesuai dengan `num_classes = 16` pada seluruh notebook.

| ID | Kategori | Ciri visual dan isi tipikal |
|---:|----------|-----------------------------|
| 0 | advertisement | Materi promosi cetak, flyer, judul komersial tebal |
| 1 | budget | Tabel keuangan, neraca, lembar buku besar |
| 2 | email | Korespondensi email tercetak dengan kepala surat (To:, From:, Subject:) |
| 3 | file_folder | Sampul indeks map, pembatas tab, lembar judul binder |
| 4 | form | Formulir standar, kotak isian, daftar periksa, kuesioner |
| 5 | handwritten | Surat, catatan jurnal, atau tanda tangan tulisan tangan |
| 6 | invoice | Tagihan, kuitansi, rincian nominal keuangan |
| 7 | letter | Surat resmi dengan alamat pengirim/penerima dan salam pembuka |
| 8 | memo | Memorandum internal (MEMORANDUM, Date:, To:, From:) |
| 9 | news_article | Kliping pers berkolom banyak, judul koran, keterangan foto |
| 10 | presentation | Slide presentasi lanskap, tata letak butir horizontal |
| 11 | questionnaire | Survei dan formulir respons berhalaman banyak |
| 12 | resume | Curriculum vitae, lamaran kerja, bagian pengalaman terstruktur |
| 13 | scientific_publication | Makalah jurnal terbit (tipografi padat dua kolom) |
| 14 | scientific_report | Laporan teknis berhalaman banyak, ringkasan lab, dokumentasi |
| 15 | specification | Skema teknik, diagram paten, spesifikasi teknis |

Bukti bahwa ke-16 kelas dipakai: `metamodel.log` mencetak `Classes: [0 1 2 ... 15]`, dan pada berkas probabilitas setiap kelas diprediksi sekitar 1.700–2.300 kali pada 32.003 baris OOF serta sekitar 415–590 kali pada 7.994 data uji (Bagian 7.2). Sebaran yang hampir merata ini konsisten dengan dataset yang seimbang (≈ 2.000 gambar per kelas pada data latih dan ≈ 500 pada data uji). Ini **dugaan** dari pola prediksi; distribusi label asli perlu dikonfirmasi dari `train.csv`.

> **Koreksi dari versi README sebelumnya:** versi awal memuat pemetaan 8 kelas sampah (`cardboard`, `electronic`, dst.) yang tidak sesuai dengan data ini. Pemetaan di atas menggantikannya.

**Catatan pasangan kelas yang mirip.** Beberapa kelas berbagi tampilan atau struktur, sehingga lazim tertukar pada klasifikasi dokumen: `form` dan `questionnaire` (4, 11), `letter` dan `memo` dan `email` (7, 8, 2), `scientific_publication` dan `scientific_report` (13, 14), serta `handwritten` dan `file_folder` (5, 3). Ini perkiraan berdasarkan deskripsi kelas, bukan temuan dari model. Matriks konfusi diperlukan untuk memastikannya.

### 3.3 Penanganan gambar rusak

Jika sebuah gambar gagal dibuka, `DocDataset` menggantinya dengan gambar hitam 224 × 224 agar pelatihan tidak berhenti. Jumlah gambar yang terkena penggantian ini tidak dicatat oleh notebook.

---

## 4. Arsitektur Stacking

```
                 ┌────────────────────┐
  gambar .tif ──►│ ConvNeXt-Base      │──► p_convnext (16)  ─┐
                 │ (5-fold, OOF)      │                      │
                 └────────────────────┘                      ├─► concat (32 fitur) ─► XGBoost ─► label
                 ┌────────────────────┐                      │   (meta-model,
  gambar .tif ──►│ DiT-Base           │──► p_dit (16)       ─┘    Optuna)
                 │ (5-fold, OOF)      │
                 └────────────────────┘
```

**Alur data:**

1. Setiap model dasar dilatih dengan Stratified 5-fold. Untuk tiap fold, model memprediksi fold validasi (mengisi `oof_*.npy`) dan seluruh data uji.
2. Prediksi uji dari kelima fold dirata-rata (`test_*.npy`).
3. Meta-model dilatih pada `[oof_dit | oof_convnext]` dengan label asli, lalu memprediksi `[test_dit | test_convnext]`.

Karena setiap baris OOF dihasilkan oleh model yang tidak pernah melihat baris tersebut saat pelatihan, meta-model tidak belajar dari prediksi yang bocor.

---

## 5. Level 0: Model Dasar

### 5.1 Praproses dan dataset

| Aspek | Nilai |
|-------|-------|
| Ukuran input | Resize ke 224 × 224 |
| Normalisasi | Rata-rata ImageNet `[0,485; 0,456; 0,406]`, simpangan baku `[0,229; 0,224; 0,225]` |
| Mode warna | Dikonversi ke RGB |
| Augmentasi | Tidak ada |
| Struktur path | `train_dir/<class_name>/<nama_file>` untuk latih, `test_dir/<nama_file>` untuk uji |

### 5.2 Model

| Model | Sumber | Inisialisasi |
|-------|--------|--------------|
| **ConvNeXt-Base** | `timm.create_model('convnext_base', pretrained=True, num_classes=16)` | Bobot pralatih, *head* baru |
| **DiT-Base** | `AutoModelForImageClassification.from_pretrained('microsoft/dit-base', num_labels=16, ignore_mismatched_sizes=True)` | Bobot pralatih (*self-supervised* pada citra dokumen), *head* baru |

Kode pada notebook juga memuat jalur `layoutlmv3` (OCR dengan EasyOCR dan `LayoutLMv3ForSequenceClassification`), tetapi **jalur ini tidak dipakai** pada stacking akhir. Meta-model hanya menerima keluaran DiT dan ConvNeXt.

### 5.3 Konfigurasi pelatihan

| Parameter | Nilai |
|-----------|-------|
| Validasi silang | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Optimizer | AdamW, `lr = 2e-5` (tanpa *scheduler* dan *weight decay* eksplisit) |
| Loss | Cross-entropy |
| Batch size | 16 |
| Epoch maksimum | 15 |
| *Early stopping* | `patience = 3` pada *validation loss* |
| Keluaran per model | `oof_<model>.npy` (32.003 × 16) dan `test_<model>.npy` (7.994 × 16) |

Notebook menyimpan berkas `oof_<model>_fold<k>.npy` setelah tiap fold selesai. Jika sesi Kaggle terputus, fold yang sudah selesai dimuat ulang dan dilewati. Dengan begitu pelatihan dapat dilanjutkan lintas sesi.

### 5.4 Jejak pelatihan fold 5 (dari log)

Kedua log hanya merekam **fold 5 dari 5**. Fold 1–4 sudah dilatih pada sesi sebelumnya, dan log menunjukkan berkas `oof_*_fold0–3.npy`, `oof_*.npy`, dan `test_*.npy` dimuat kembali dari dataset checkpoint lalu fold tersebut dilewati ("Fold 1–4 sudah selesai sebelumnya, melewati..."). Kurva pelatihan fold 1–4 tidak tersedia.

| Epoch | ConvNeXt: loss latih | ConvNeXt: loss validasi | DiT: loss latih | DiT: loss validasi |
|------:|------:|------:|------:|------:|
| 1 | 0,9925 | 0,6962 | 1,9942 | 1,3123 |
| 2 | 0,5073 | 0,6301 | 1,1858 | 1,0402 |
| 3 | 0,2893 | **0,5989** | 0,9109 | 0,8945 |
| 4 | 0,1296 | 0,6287 | 0,7177 | 0,8416 |
| 5 | 0,0552 | 0,6687 | 0,5518 | **0,7867** |
| 6 | 0,0321 | 0,7527 *(berhenti)* | 0,4191 | 0,8171 |
| 7 | | | 0,3212 | 0,8451 |
| 8 | | | 0,2452 | 0,9404 *(berhenti)* |

Cetak tebal menandai loss validasi terendah (epoch terbaik). *Early stopping* dengan `patience = 3` menghentikan ConvNeXt pada epoch 6 dan DiT pada epoch 8.

**Interpretasi:**

1. **Kedua model *overfit* setelah epoch terbaik.** Loss latih terus turun (ConvNeXt hingga 0,03, DiT hingga 0,25) sementara loss validasi naik kembali. ConvNeXt mencapai optimum sangat dini (epoch 3), dan loss latih yang sangat kecil (0,03) menjelaskan mengapa probabilitasnya sangat yakin (Bagian 7.3).
2. **ConvNeXt lebih unggul pada fold ini**: loss validasi terbaik 0,599 dibanding 0,787 untuk DiT, sejalan dengan keyakinan DiT yang lebih rendah pada data OOF. Ini hanya satu fold, sehingga belum bisa digeneralisasi.
3. **Loss validasi pada epoch akhir jauh dari epoch terbaik**: 0,7527 vs 0,5989 untuk ConvNeXt dan 0,9404 vs 0,7867 untuk DiT. Selisih ini penting karena kemungkinan bobot yang dipakai untuk prediksi adalah bobot epoch akhir (lihat Bagian 8.2).
4. **Biaya komputasi**: sekitar 1.600 iterasi per epoch (batch 16, ≈ 25.600 gambar latih per fold). ConvNeXt memerlukan sekitar 26 menit per epoch dan total ≈ 10.000 detik (≈ 2,8 jam) untuk satu fold. DiT memerlukan sekitar 21 menit per epoch dan total ≈ 10.700 detik (≈ 3,0 jam). Lima fold penuh berarti sekitar 14–15 jam per model, yang menjelaskan mengapa pelatihan dipecah lintas sesi dan memakai checkpoint.
5. **Log DiT** menunjukkan `BeitForImageClassification` dimuat dari `microsoft/dit-base`: bobot `classifier` dan `beit.pooler.layernorm` diinisialisasi baru ("MISSING"), sedangkan `lm_head` dan `layernorm` dari pralatih tidak dipakai ("UNEXPECTED"). Hal ini wajar untuk transfer ke tugas klasifikasi.
6. Peringatan `Corrupt EXIF data` dari PIL muncul pada pembacaan berkas TIFF. Peringatan ini tidak menghentikan pembacaan gambar, tetapi Python menampilkan peringatan yang sama hanya sekali per lokasi kode, jadi jumlah gambar terdampak tidak dapat dihitung dari log.

---

## 6. Level 1: Meta-Model

### 6.1 Fitur dan label

`X_meta = concat([oof_dit, oof_convnext], axis=1)` dengan bentuk (32.003, 32), dan `y_meta` adalah label asli dari `train.csv`.

### 6.2 Meta-model dasar

XGBoost dengan konfigurasi awal: `n_estimators=500`, `max_depth=6`, `learning_rate=0,05`, `subsample=0,8`, `colsample_bytree=0,8`, `objective=multi:softprob`, `tree_method=hist`, `random_state=42`.

### 6.3 Penyetelan hiperparameter dengan Optuna

| Pengaturan | Nilai |
|------------|-------|
| Sampler | `TPESampler(seed=42, n_startup_trials=10)` |
| Jumlah trial | 30 |
| Validasi | `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` pada `X_meta` |
| Objektif | Memaksimalkan rata-rata **akurasi** CV |
| Penyimpanan study | SQLite (`optuna_xgb_accuracy.db`), `load_if_exists=True` sehingga dapat dilanjutkan |

**Ruang pencarian:**

| Hiperparameter | Rentang |
|----------------|---------|
| `n_estimators` | 300–1200 (langkah 100) |
| `max_depth` | 2–9 |
| `learning_rate` | 0,01–0,2 (skala log) |
| `min_child_weight` | 1–12 (skala log) |
| `subsample` | 0,65–1,0 |
| `colsample_bytree` | 0,6–1,0 |
| `gamma` | 0–5 |
| `reg_alpha` | 1e-8–10 (skala log) |
| `reg_lambda` | 1e-2–100 (skala log) |
| `max_bin` | 128–512 (langkah 64) |

Parameter terbaik dipakai untuk melatih ulang XGBoost pada **seluruh** `X_meta`, kemudian model disimpan sebagai `xgb_meta_optuna.joblib`.

### 6.4 Penyelarasan urutan pada data uji

Prediksi uji model dasar disusun mengikuti urutan `os.listdir`, sedangkan submission harus mengikuti urutan `sample_submission.csv`. Notebook memetakan `id + ".tif"` ke indeks baris array uji (`row_indices`) dan mengurutkan ulang kedua array sebelum digabung. Langkah ini dilindungi *assert* (semua nama berkas ditemukan, nama berkas unik, bentuk akhir `(7994, 32)`). Kesalahan urutan di sini akan menurunkan skor secara drastis tanpa pesan galat, sehingga pemeriksaan tersebut penting.

### 6.5 Validasi submission

Sebelum disimpan, submission diperiksa: 7.994 baris, label dalam rentang 0–15, dan tidak ada nilai kosong.

### 6.6 Perbandingan kandidat meta-model (dari `metamodel.log`)

Log `metamodel.log` memperlihatkan langkah yang **tidak ada** pada `metamodel.ipynb` yang dilampirkan: empat kandidat meta-model dievaluasi dengan CV 5-fold pada `X_meta` (32.003 × 32), lalu model terbaik dipakai membuat `submission_initial.csv`. Notebook yang dilampirkan hanya memuat XGBoost dan Optuna, jadi log ini berasal dari versi notebook yang berbeda. Konfigurasi tiap kandidat tidak terlihat dari log.

| Meta-model | Akurasi (rata-rata ± simpangan baku antarfold) | Macro F1 | Log loss | Waktu per fold |
|------------|-----------------------:|---------:|---------:|---------------:|
| Logistic Regression | 0,8411 ± 0,0064 | 0,8413 ± 0,0064 | 0,6071 ± 0,0223 | ≈ 6 dtk |
| Random Forest | 0,8342 ± 0,0070 | 0,8342 ± 0,0072 | 0,9494 ± 0,0225 | ≈ 37 dtk |
| **XGBoost** | **0,8418 ± 0,0068** | **0,8418 ± 0,0063** | **0,5417 ± 0,0193** | ≈ 24 dtk |
| LightGBM | 0,8386 ± 0,0060 | 0,8387 ± 0,0058 | 0,5523 ± 0,0200 | ≈ 15 dtk |

Log mencetak `Best model: XGBoost`; kriteria pemilihan tidak terlihat, tetapi XGBoost memang terbaik pada Macro F1 dan log loss.

Akurasi per fold (fold 1–5):

| Meta-model | F1 | F2 | F3 | F4 | F5 |
|------------|---:|---:|---:|---:|---:|
| Logistic Regression | 0,8416 | 0,8355 | 0,8485 | 0,8461 | 0,8339 |
| Random Forest | 0,8357 | 0,8271 | 0,8427 | 0,8386 | 0,8272 |
| XGBoost | 0,8378 | 0,8342 | 0,8500 | 0,8480 | 0,8389 |
| LightGBM | 0,8352 | 0,8336 | 0,8469 | 0,8430 | 0,8342 |

**Interpretasi:**

1. **Perbedaan akurasi antar meta-model sangat kecil.** Selisih terbesar antara empat kandidat hanya 0,0076 (XGBoost vs Random Forest), jauh di bawah simpangan baku antarfold (≈ 0,006–0,007). Antara XGBoost dan Logistic Regression, selisih akurasi rata-rata hanya 0,0007 dan arahnya berganti-ganti antarfold (LR unggul pada fold 1 dan 2, XGBoost pada fold 3–5), sehingga keduanya tidak dapat dibedakan pada akurasi.
2. **XGBoost unggul konsisten pada log loss**: lebih rendah dari Logistic Regression di kelima fold (selisih rata-rata 0,065). Ini berarti probabilitas XGBoost lebih terkalibrasi, yang relevan bila probabilitas dipakai selain untuk `argmax`.
3. **Random Forest paling lemah**, terutama pada log loss (0,949), tanda probabilitas yang kurang terkalibrasi pada 32 fitur probabilitas.
4. **Pola kesulitan fold sama pada semua model**: fold 3 dan 4 paling mudah, fold 2 dan 5 paling sulit. Karena semua model mengalami pola yang sama pada pembagian fold yang sama, perbedaan antarfold berasal dari data, bukan dari model.
5. **Pembanding terhadap model dasar:** log loss meta-model terbaik (0,542) lebih rendah daripada loss validasi terbaik ConvNeXt pada fold 5 (0,599) dan DiT (0,787). Karena fold dan prosedur berbeda, ini hanya indikasi bahwa stacking memperbaiki kualitas probabilitas, bukan perbandingan yang ketat.
6. **Pilihan model sederhana:** karena Logistic Regression hampir sama akurat dan jauh lebih murah serta lebih mudah dijelaskan, nilai tambah XGBoost dan penyetelan Optuna sebaiknya diuji secara hati-hati (lihat Bagian 9).

**Distribusi prediksi `submission_initial.csv` (XGBoost):** 461–548 prediksi per kelas (rata-rata 499,6; simpangan baku 25,2), `presentation` (10) terbanyak (548) dan `news_article` (9) tersedikit (461). Sebagai pembanding, rata-rata sederhana ConvNeXt + DiT menghasilkan simpangan baku 35,4 (`questionnaire` hanya 425 dan `presentation` sebanyak 553). Stacking membuat sebaran lebih merata dan lebih dekat ke ≈ 500 per kelas, yang konsisten dengan dugaan data uji seimbang. Ini belum membuktikan akurasi yang lebih tinggi.

---

## 7. Hasil

### 7.1 Verifikasi berkas probabilitas

| Pemeriksaan | ConvNeXt-Base | DiT-Base |
|-------------|---------------|----------|
| Bentuk OOF | (32.003, 16) | (32.003, 16) |
| Bentuk prediksi uji | (7.994, 16) | (7.994, 16) |
| Jumlah probabilitas per baris | 1,0 (selisih < 3e-7) | 1,0 (selisih < 3e-7) |
| Baris nol (fold terlewat) | 0 | 0 |
| Nilai NaN | 0 | 0 |
| Ukuran fold 0–4 | 6.401 / 6.401 / 6.401 / 6.400 / 6.400 | 6.401 / 6.401 / 6.401 / 6.400 / 6.400 |
| Jumlah probabilitas per baris uji | 1,0 (selisih < 2e-7) | 1,0 (selisih < 2e-7) |

Prediksi uji adalah rata-rata kelima fold (tiap fold menyumbang 1/5), jadi jumlah probabilitas per baris uji bernilai 1,0 hanya bila **kelima fold ikut menyumbang**. Hasil ini membuktikan tidak ada fold yang terlewat saat sesi dilanjutkan dari checkpoint. Seluruh baris OOF terisi (tidak ada baris nol) dan jumlah ukuran kelima fold sama dengan 32.003, sehingga kelima fold lengkap dan tergabung dengan benar pada `oof_*.npy`. Ukuran fold sesuai pembagian `StratifiedKFold` 5-fold pada 32.003 baris.

### 7.2 Distribusi kelas prediksi (argmax)

| Kelas | ConvNeXt OOF | DiT OOF | ConvNeXt uji | DiT uji |
|------:|-------------:|--------:|-------------:|--------:|
| 0 (advertisement) | 2.022 | 2.143 | 504 | 523 |
| 1 (budget) | 1.915 | 1.947 | 476 | 478 |
| 2 (email) | 2.008 | 2.020 | 501 | 502 |
| 3 (file_folder) | 2.092 | 2.177 | 522 | 560 |
| 4 (form) | 2.128 | 2.184 | 550 | 559 |
| 5 (handwritten) | 2.125 | 1.969 | 525 | 487 |
| 6 (invoice) | 2.057 | 2.100 | 503 | 533 |
| 7 (letter) | 2.122 | 2.042 | 533 | 517 |
| 8 (memo) | 2.036 | 1.857 | 487 | 473 |
| 9 (news_article) | 1.916 | 1.988 | 479 | 487 |
| 10 (presentation) | **2.338** | 2.008 | **587** | 493 |
| 11 (questionnaire) | 1.845 | 1.762 | 440 | 415 |
| 12 (resume) | 1.897 | 1.906 | 489 | 473 |
| 13 (scientific_publication) | 1.988 | 1.989 | 500 | 497 |
| 14 (scientific_report) | **1.668** | 2.107 | **429** | 537 |
| 15 (specification) | 1.846 | 1.804 | 469 | 460 |

**Interpretasi:** semua kelas dipakai oleh kedua model. Penyimpangan yang paling menonjol ada pada ConvNeXt: `presentation` (10) diprediksi 2.338 kali (lebih banyak dari rata-rata ≈ 2.000) dan `scientific_report` (14) hanya 1.668 kali, sedangkan DiT memprediksi `scientific_report` sebanyak 2.107 kali. **Hipotesis:** ConvNeXt mengklasifikasikan sebagian `scientific_report` sebagai `presentation`, dan DiT tidak. Hipotesis ini perlu diuji dengan matriks konfusi, dan perlu dicatat bahwa kedua kelas tampak berbeda (lanskap berbutir vs teks padat), sehingga penyebabnya bisa saja resolusi 224 × 224 yang membuat teks halaman padat terlihat seperti slide. Kedua model juga memprediksi `questionnaire` (11) lebih jarang dari rata-rata (1.845 dan 1.762), yang dapat berarti sebagian jatuh ke `form` (4). Pola serupa pada data uji (587 dan 429) menunjukkan perilaku yang sama pada data yang belum terlihat. Perbedaan seperti ini menguntungkan stacking karena meta-model dapat belajar mengoreksinya.

### 7.3 Keyakinan (*confidence*) model

| Ukuran | ConvNeXt OOF | DiT OOF | ConvNeXt uji | DiT uji |
|--------|-------------:|--------:|-------------:|--------:|
| Rata-rata probabilitas maksimum | 0,923 | 0,890 | 0,874 | 0,829 |
| Proporsi baris dengan p maks > 0,9 | 79,6% | 70,8% | 67,6% | 57,1% |
| Proporsi baris dengan p maks < 0,5 | 4,1% | 6,1% | 8,4% | 11,6% |
| Rata-rata entropi (OOF) | 0,233 | 0,334 | | |

Rata-rata probabilitas maksimum per fold (OOF): ConvNeXt 0,919 / 0,919 / 0,928 / 0,932 / 0,919 dan DiT 0,877 / 0,875 / 0,895 / 0,906 / 0,898 untuk fold 0–4.

**Interpretasi:**

1. **ConvNeXt lebih yakin dan DiT lebih berhati-hati** pada data OOF (0,923 vs 0,890; entropi 0,233 vs 0,334).
2. **Variasi antarfold kecil** (sekitar 0,013 untuk ConvNeXt dan 0,031 untuk DiT), tanda pelatihan cukup stabil. Fold 3 paling yakin untuk kedua model, dan fold 0–1 paling kurang yakin pada DiT.
3. **Keyakinan pada data uji lebih rendah daripada OOF** (ConvNeXt 0,874 vs 0,923, DiT 0,829 vs 0,890). Lihat Bagian 8.5 untuk akibatnya pada meta-model.

### 7.4 Kesepakatan antarmodel

| Ukuran | OOF | Uji |
|--------|----:|----:|
| Kesepakatan argmax ConvNeXt vs DiT | 79,4% | 85,7% |
| Korelasi probabilitas maksimum kedua model (OOF) | 0,396 | |
| Rata-rata p maks saat **sepakat** (ConvNeXt / DiT) | 0,960 / 0,937 | |
| Rata-rata p maks saat **tidak sepakat** (ConvNeXt / DiT) | 0,780 / 0,709 | |

**Interpretasi:**

- Kedua model berbeda pendapat pada **20,6% baris OOF**. Pada baris tersebut paling banyak satu model yang benar, sehingga ada ruang nyata bagi meta-model untuk memilih model yang tepat.
- Keyakinan turun tajam saat model tidak sepakat (0,78 dan 0,71 dibanding 0,96 dan 0,94). Artinya probabilitas membawa sinyal tentang baris yang sulit, dan inilah yang dimanfaatkan XGBoost sebagai fitur.
- Korelasi keyakinan hanya 0,40, jadi kedua model tidak selalu ragu pada gambar yang sama. Ini mendukung alasan memakai dua arsitektur yang berbeda.
- Kesepakatan pada data uji lebih tinggi (85,7%) karena prediksi uji adalah rata-rata lima model fold, yang cenderung lebih stabil.

### 7.5 Perkiraan performa model dasar (bukan hasil pengukuran)

Akurasi model dasar tidak dapat dihitung tanpa `train.csv`, tetapi log memberi batas atas yang berguna. Pada `metamodel.log`, meta-model terbaik mencapai **akurasi CV ≈ 0,842**. Stacking yang berfungsi normal hampir selalu setidaknya sebaik model dasar terbaiknya, sehingga akurasi OOF ConvNeXt dan DiT kemungkinan **tidak lebih tinggi dari sekitar 0,84**.

Bandingkan dengan rata-rata probabilitas maksimum pada Bagian 7.3 (ConvNeXt 0,923 dan DiT 0,890). Bila probabilitas terkalibrasi, angka itu seharusnya mendekati akurasi. Selisih sekitar 5–8 poin ke atas menunjukkan bahwa **kedua model terlalu yakin (*overconfident*)**, terutama ConvNeXt. Hal ini konsisten dengan jejak pelatihan pada Bagian 5.4 (loss latih sangat kecil, loss validasi naik setelah epoch terbaik). Kesimpulan ini adalah dugaan dari bukti tidak langsung. Kepastian memerlukan akurasi dan kurva keandalan dari label asli.

**Implikasi:** probabilitas dasar yang terlalu yakin memang membuat meta-model perlu belajar "menurunkan kepercayaan", dan ini salah satu alasan stacking memperbaiki log loss.

### 7.6 Metrik terhadap label asli

| Model | Akurasi OOF | Macro F1 OOF | Log loss OOF |
|-------|------------:|-------------:|-------------:|
| ConvNeXt-Base (level 0) | `[isi]` | `[isi]` | `[isi]` |
| DiT-Base (level 0) | `[isi]` | `[isi]` | `[isi]` |
| Rata-rata sederhana ConvNeXt + DiT | `[isi]` | `[isi]` | `[isi]` |
| XGBoost meta-model awal (CV 5-fold) | `[isi]` | `[isi]` | `[isi]` |
| *Kandidat dari log:* Logistic Regression (CV 5-fold) | 0,8411 | 0,8413 | 0,6071 |
| *Kandidat dari log:* **XGBoost** (CV 5-fold, konfigurasi tidak terlihat) | **0,8418** | **0,8418** | **0,5417** |
| *Kandidat dari log:* LightGBM (CV 5-fold) | 0,8386 | 0,8387 | 0,5523 |
| *Kandidat dari log:* Random Forest (CV 5-fold) | 0,8342 | 0,8342 | 0,9494 |
| **XGBoost meta-model + Optuna (CV 5-fold)** | `[isi]` | `[isi]` | `[isi]` |
| Skor leaderboard publik | `[isi]` | | |

**Parameter terbaik Optuna:** `[isi dari study.best_params]`

Perbandingan tiga baris pertama dengan baris meta-model menjawab pertanyaan inti: apakah stacking benar-benar lebih baik daripada model tunggal dan rata-rata sederhana. Rata-rata sederhana kedua model dapat dihitung langsung dari berkas `.npy` bila `train.csv` tersedia (`((oof_dit + oof_convnext) / 2).argmax(1)`).

---

## 8. Diskusi: Validitas dan Keterbatasan

### 8.1 Optimisme pada skor CV meta-model

Hiperparameter dipilih berdasarkan akurasi CV pada `X_meta`, lalu skor terbaik itu juga yang dilaporkan. Pemilihan terbaik dari 30 trial menimbulkan bias ke atas. Pada 32 fitur dan 32.003 sampel biasanya kecil, tetapi besarnya tidak terukur. Estimasi yang lebih jujur memerlukan CV bersarang atau *hold-out* yang tidak dipakai untuk tuning.

### 8.2 Pemilihan epoch dan potensi cacat pada *checkpoint* terbaik

Pada kode *early stopping*, `best_model_state = model.state_dict()` menyimpan **referensi** ke tensor model, bukan salinan. Tensor tersebut terus berubah selama pelatihan, sehingga bobot yang dimuat kembali kemungkinan besar adalah bobot **epoch terakhir**, bukan epoch dengan *validation loss* terendah. Variabel `val_probs` juga ditimpa tiap epoch, jadi probabilitas OOF berasal dari epoch terakhir yang dijalankan. Dampaknya: *early stopping* hanya berfungsi sebagai penghenti, bukan pemilih model terbaik. Perbaikan: `copy.deepcopy(model.state_dict())` dan simpan `val_probs` pada epoch terbaik.

**Log fold 5 menunjukkan bahwa selisihnya tidak kecil** (Bagian 5.4). ConvNeXt mencapai loss validasi terbaik 0,5989 pada epoch 3, tetapi berhenti pada epoch 6 dengan loss 0,7527. DiT terbaik pada epoch 5 (0,7867) dan berhenti pada epoch 8 (0,9404). Bila dugaan di atas benar, OOF dan prediksi uji fold 5 berasal dari model yang jauh lebih *overfit* daripada model terbaik, dan ini sejalan dengan probabilitas yang terlalu yakin pada Bagian 7.5. **Cara memverifikasi** dengan `train.csv`: hitung log loss dari `oof_convnext_fold4.npy` (fold terakhir) terhadap label fold tersebut. Hasil ≈ 0,7527 berarti bobot epoch 6 dipakai, sedangkan ≈ 0,5989 berarti bobot epoch 3 dipakai. Lakukan hal yang sama untuk DiT (0,9404 vs 0,7867).

### 8.3 Pemilihan epoch memakai fold validasi yang sama dengan OOF

*Early stopping* memakai fold validasi untuk menghentikan pelatihan, dan fold yang sama dipakai sebagai prediksi OOF. Ini membuat OOF sedikit optimistis relatif terhadap data yang benar-benar baru. Efek ini biasanya kecil, tetapi meta-model dilatih pada OOF tersebut dan dapat sedikit terlalu percaya pada model dasar.

### 8.4 Pembagian fold untuk model dasar dan CV meta-model berbeda

Model dasar memakai `random_state=42` dan CV Optuna memakai `random_state=2026`. Ini tidak menyebabkan kebocoran karena OOF sudah murni *out-of-fold*, tetapi pada tiap fold CV meta-model, sebagian fitur meta berasal dari model yang pernah dilatih pada baris validasi meta tersebut. Efeknya adalah ciri standar stacking dan umumnya diterima, namun perlu diketahui saat menafsirkan skor CV.

### 8.5 Ketidaksesuaian distribusi probabilitas OOF dan uji

Meta-model dilatih pada probabilitas OOF yang berasal dari **satu model per baris**, tetapi diterapkan pada probabilitas uji yang merupakan **rata-rata lima model fold**. Berkas probabilitas menunjukkan perbedaannya: rata-rata probabilitas maksimum turun dari 0,923 (OOF) menjadi 0,874 (uji) untuk ConvNeXt, dan dari 0,890 menjadi 0,829 untuk DiT. Proporsi baris dengan keyakinan di atas 0,9 turun dari 79,6% menjadi 67,6% (ConvNeXt) dan dari 70,8% menjadi 57,1% (DiT).

Ada dua penyebab yang mungkin dan keduanya tidak dapat dibedakan dari berkas yang ada:

1. **Efek rata-rata fold.** Merata-ratakan probabilitas dari model yang tidak sepakat menurunkan nilai maksimum. Ini adalah ciri standar stacking dan umumnya tidak berbahaya, tetapi XGBoost menetapkan ambang pada nilai probabilitas, sehingga pergeseran skala ini dapat memengaruhi keputusan.
2. **Pergeseran distribusi** (*distribution shift*) antara data latih dan uji, yaitu data uji memang lebih sulit atau berbeda.

Karena prediksi uji per fold tidak disimpan, kedua kemungkinan tidak dapat dipisahkan. Jika penyebab (1) dominan, pergeseran ini tidak merusak hasil. Jika penyebab (2) dominan, skor CV akan terlalu optimistis dibanding skor leaderboard.

### 8.6 Keterbatasan lain

- **Tanpa augmentasi** dan **tanpa *learning rate scheduler***, padahal dokumen dapat bervariasi dalam rotasi, kontras, dan resolusi.
- **Resize 224 × 224** pada gambar dokumen dapat menghilangkan detail halus. Resolusi lebih tinggi mungkin membantu, tetapi menaikkan biaya komputasi.
- **Analisis kesalahan belum ada**: tidak ada matriks konfusi, laporan klasifikasi per kelas, maupun distribusi kelas pada data latih.
- **Keragaman model dasar terbatas**: hanya dua arsitektur. Keragaman lebih besar (misalnya model berbasis teks atau OCR seperti LayoutLMv3, yang sudah ada di kode tetapi tidak dipakai) berpotensi meningkatkan stacking.
- **Meta-model hanya memakai probabilitas**, tanpa fitur tambahan seperti ukuran gambar atau jumlah halaman.
- **Seed tunggal**: tidak ada pengulangan dengan seed berbeda, sehingga selisih kecil antarkonfigurasi tidak dapat dinyatakan signifikan.
- **Log hanya mencakup fold 5** untuk kedua model dasar, dan tidak ada kurva latih untuk fold 1–4. Kualitas fold lain tidak dapat dinilai dari log.
- **Versi notebook meta-model tidak konsisten dengan log.** `metamodel.log` menampilkan perbandingan empat kandidat dan keluaran `submission_initial.csv`, sedangkan `metamodel.ipynb` memuat XGBoost, Optuna, dan `submission_optuna.csv`. Hasil Optuna tidak ada di log, sehingga manfaat penyetelan terhadap XGBoost awal belum terdokumentasi.
- **Perbedaan antar meta-model sangat kecil** (Bagian 6.6) dan lebih kecil daripada variasi antarfold, sehingga memilih XGBoost daripada Logistic Regression berdasarkan akurasi tidak dapat dipertahankan secara statistik.

---

## 9. Rekomendasi Tindak Lanjut

Disusun menurut dampak terhadap kesahihan hasil:

1. **Perbaiki penyimpanan model terbaik** (`deepcopy`) dan simpan OOF dari epoch terbaik (lihat 8.2).
2. **Hitung metrik terhadap label asli** (Bagian 7.6) dari berkas `.npy` yang sudah ada dan `train.csv`, termasuk kalibrasi probabilitas (kurva keandalan dan ECE).
3. **Laporkan matriks konfusi** untuk memeriksa hipotesis bahwa ConvNeXt menukar `scientific_report` (14) dengan `presentation` (10) (Bagian 7.2), serta pasangan mirip pada Bagian 3.2 seperti `form` dan `questionnaire`.
4. **Bandingkan dengan baseline sederhana**: rata-rata probabilitas dan regresi logistik sebagai meta-model. Log sudah menunjukkan bahwa Logistic Regression (akurasi 0,8411) hampir sama dengan XGBoost (0,8418). Uji juga Logistic Regression pada *logit* (log-probabilitas) bukan probabilitas mentah, dan **hitung tuning Optuna terhadap metrik yang sama dengan metrik pembanding** agar manfaatnya terukur.
5. **Pisahkan himpunan tuning dan evaluasi** atau gunakan CV bersarang untuk estimasi yang tidak bias.
6. **Tambahkan augmentasi dan scheduler** (misalnya *cosine decay* dengan *warmup*) pada model dasar.
7. **Tambahkan model dasar ketiga yang beragam**, seperti LayoutLMv3 yang sudah tersedia di kode, atau ConvNeXt/ViT pada resolusi lebih tinggi. Pasangan kelas yang mirip secara tata letak tetapi berbeda isi teks (`letter`/`memo`/`email`, `form`/`questionnaire`, `scientific_publication`/`scientific_report`) adalah kandidat utama yang dapat terbantu oleh model berbasis teks, dan resolusi di atas 224 × 224 membantu membaca teks pada dokumen padat.
8. **Gunakan metrik yang sama dengan metrik kompetisi** sebagai objektif Optuna bila berbeda dari akurasi (misalnya Macro F1).
9. **Kurangi *overfit* model dasar**: ConvNeXt mencapai optimum pada epoch 3 dan DiT pada epoch 5 sementara loss latih turun terus. Pertimbangkan *weight decay*, augmentasi, *label smoothing*, *scheduler*, dan penyimpanan bobot epoch terbaik. *Label smoothing* atau *temperature scaling* juga mengurangi keyakinan berlebih yang tampak pada Bagian 7.5.
10. **Simpan prediksi uji per fold** (bukan hanya rata-ratanya) agar pergeseran distribusi pada Bagian 8.5 dapat didiagnosis.
11. **Simpan log pelatihan semua fold** (bukan hanya fold terakhir) agar kurva loss tiap fold dapat dianalisis.
12. **Ulangi dengan beberapa seed** dan laporkan rata-rata ± simpangan baku.

---

## 10. Struktur Repositori dan Reproduksibilitas

```
.
├── notebooks/
│   ├── convnext.ipynb      # Level 0: ConvNeXt-Base, 5-fold, OOF + prediksi uji
│   ├── dit-base.ipynb      # Level 0: DiT-Base, 5-fold, OOF + prediksi uji
│   └── metamodel.ipynb     # Level 1: XGBoost + Optuna, submission
├── level1_outputs/
│   ├── oof_convnext.npy        # (32003, 16) probabilitas OOF
│   ├── oof_convnext_fold{0-4}.npy   # OOF per fold (6401/6401/6401/6400/6400 baris)
│   ├── test_convnext.npy       # (7994, 16) rata-rata 5 fold
│   ├── oof_dit.npy             # (32003, 16)
│   ├── oof_dit_fold{0-4}.npy
│   └── test_dit.npy            # (7994, 16)
└── README.md
```

**Lingkungan.** Notebook dijalankan di Kaggle dengan data kompetisi `document-image-classification-sd234504`. Metadata notebook tidak konsisten: `convnext.ipynb` mencatat Python 3.10, `dit-base.ipynb` mencatat Python 3.6.6 dengan akselerator `none`. **Log eksekusi menunjukkan semua notebook sebenarnya berjalan pada Python 3.13** (path `/usr/local/lib/python3.13/dist-packages/...`). Metadata notebook sebaiknya diabaikan, dan versi pustaka perlu dicatat dari lingkungan sebenarnya. Pelatihan DiT dan ConvNeXt memerlukan GPU. Estimasi waktu: ≈ 2,8 jam (ConvNeXt) dan ≈ 3,0 jam (DiT) per fold, sedangkan evaluasi empat kandidat meta-model selesai dalam sekitar 7,5 menit (430 detik) ditambah ≈ 25 detik untuk pembuatan submission.

**Pustaka utama:** `torch`, `torchvision`, `timm`, `transformers`, `scikit-learn`, `xgboost`, `optuna`, `joblib`, `pandas`, `numpy`, `Pillow`, `tqdm` (serta `easyocr` untuk jalur LayoutLMv3 yang tidak dipakai).

**Urutan eksekusi:**

1. Jalankan `convnext.ipynb` dengan `MODEL_TO_TRAIN = 'convnext'`. Hasil: `oof_convnext.npy` dan `test_convnext.npy`.
2. Jalankan `dit-base.ipynb` dengan `MODEL_TO_TRAIN = 'dit'`. Hasil: `oof_dit.npy` dan `test_dit.npy`.
3. Unggah keempat berkas `.npy` sebagai dataset Kaggle (di notebook dipanggil `dataset-level-1`), dan siapkan juga `output.csv` yang berisi daftar `filename` data uji sesuai urutan inferensi (dataset `urutan-test-df`).
4. Jalankan `metamodel.ipynb`. Sel Optuna menyimpan study ke SQLite sehingga dapat dilanjutkan. Hasil akhir: `submission_optuna.csv`.

**Catatan penting:** urutan baris `test_*.npy` mengikuti `os.listdir(test_dir)`, yang tidak dijamin sama antar-sesi atau antar-mesin. Karena itu urutan tersebut disimpan di `output.csv` dan dipakai pada penyelarasan di meta-model. Jangan menjalankan ulang inferensi uji tanpa menyimpan ulang urutan berkas.

**Reproduksibilitas:** `random_state=42` dipakai pada pembagian fold model dasar, sampler Optuna, dan XGBoost; CV Optuna memakai `2026`. Hasil dapat bergeser sedikit akibat nondeterminisme GPU, `n_jobs=-1`, dan versi pustaka. Inisialisasi *head* baru pada model dasar tidak diberi *seed* eksplisit.

---

*Pernyataan bertanda "hipotesis", "kemungkinan", atau "dugaan" adalah interpretasi yang belum diuji oleh eksperimen dalam notebook. Nilai bertanda `[isi]` perlu diganti dengan keluaran eksekusi Anda.*
