# Klasifikasi Tingkat Kerusakan Struktural Bangunan Pascagempa

**Laporan Interpretasi Ilmiah — Analisis Data Eksploratif (EDA) dan Pemodelan**

> Dokumen ini menginterpretasikan hasil dari dua notebook: `01_eda.ipynb` (eksplorasi data) dan `02_modeling.ipynb` (rekayasa fitur, pelatihan model, dan evaluasi). Seluruh angka dalam laporan ini diambil langsung dari keluaran sel notebook. Hal yang **tidak** dapat dibuktikan oleh keluaran notebook ditandai sebagai *hipotesis* atau *keterbatasan*, bukan sebagai temuan.

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Latar Belakang dan Tujuan](#2-latar-belakang-dan-tujuan)
3. [Data](#3-data)
4. [Hasil EDA dan Interpretasinya](#4-hasil-eda-dan-interpretasinya)
5. [Metodologi Pemodelan](#5-metodologi-pemodelan)
6. [Hasil Pemodelan dan Interpretasinya](#6-hasil-pemodelan-dan-interpretasinya)
7. [Diskusi: Validitas dan Keterbatasan](#7-diskusi-validitas-dan-keterbatasan)
8. [Kesimpulan](#8-kesimpulan)
9. [Rekomendasi Tindak Lanjut](#9-rekomendasi-tindak-lanjut)
10. [Struktur Repositori dan Reproduksibilitas](#10-struktur-repositori-dan-reproduksibilitas)

---

## 1. Ringkasan Eksekutif

Penelitian ini memprediksi **tingkat kerusakan bangunan (Grade 1–5)** dari atribut struktural dan lokasi bangunan pada data survei pascagempa (kompetisi *structural-damage-classification*). Data latih berisi 533.465 bangunan; data uji kompetisi berisi 228.629 bangunan.

Temuan utama:

| # | Temuan | Bukti |
|---|--------|-------|
| 1 | Data bersih secara formal (tanpa nilai hilang dan tanpa duplikat), tetapi memuat anomali `age_building = 999` pada 2.198 baris (0,41%). | EDA sel 4, 5, 15 |
| 2 | Kelas target **tidak seimbang** dan terkonsentrasi pada kerusakan berat (Grade 5 = 36,2%; Grade 1 = 10,3%). | EDA sel 16 |
| 3 | Asosiasi terkuat dengan `damage_grade` berasal dari **variabel lokasi** (ward 0,447; kecamatan/vdcmun 0,369; distrik 0,301), lalu **material struktur** (superstructure 0,300; roof 0,287; foundation 0,256). Variabel geometri berasosiasi lebih lemah. | EDA sel 14 |
| 4 | Model terbaik adalah *ensemble* tiga model (LightGBM, XGBoost, KNN spasial berbasis urutan survei) dengan **Macro F1 = 0,594** dan akurasi 0,62 pada 106.254 sampel validasi. | Modeling sel 7 |
| 5 | Kelas **Grade 2 dan Grade 3** paling sulit dibedakan (F1 masing-masing 0,46 dan 0,49), sedangkan Grade 5 dan Grade 1 paling mudah (0,75 dan 0,70). | Modeling sel 7 |

**Catatan kehati-hatian utama:** skor 0,594 diperoleh setelah bobot kelas dan bobot *ensemble* dioptimalkan pada himpunan validasi yang sama dengan yang dipakai untuk melaporkan skor, sehingga bersifat optimistis. Estimasi yang lebih konservatif adalah skor model tunggal tanpa tuning, yaitu **0,584** (LightGBM). Selain itu, sebagian besar daya prediksi berasal dari struktur *building_id* (urutan survei), yang membatasi generalisasi ke gempa atau wilayah lain (lihat Bagian 7).

---

## 2. Latar Belakang dan Tujuan

Penilaian cepat kerusakan bangunan setelah gempa sangat menentukan prioritas bantuan, inspeksi, dan rekonstruksi. Survei lapangan berskala ratusan ribu bangunan membutuhkan waktu lama, sehingga model prediktif yang memanfaatkan atribut bangunan dapat membantu memprioritaskan inspeksi.

**Tujuan:**

1. Mengkarakterisasi data survei melalui EDA (kualitas data, distribusi, dan hubungan antarvariabel).
2. Membangun model klasifikasi lima kelas untuk `damage_grade`.
3. Mengevaluasi performa dengan metrik yang sesuai untuk data tak seimbang (**Macro F1**).

---

## 3. Data

### 3.1 Sumber dan skala

| Berkas | Ukuran | Keterangan |
|--------|--------|------------|
| `train.csv` | 533.465 baris × 17 kolom | Memiliki label `damage_grade` |
| `test.csv` | 228.629 baris | Tanpa label (kompetisi) |

Setelah menghapus baris `age_building = 999`, data latih menjadi **531.267 baris**.

### 3.2 Variabel

| Kelompok | Variabel |
|----------|----------|
| Identitas dan wilayah | `building_id`, `district_id` (11 nilai), `vdcmun_id` (110 nilai), `ward_id` (945 nilai) |
| Geometri / numerik | `count_floors_pre_eq`, `age_building`, `plinth_area_sq_ft`, `height_ft_pre_eq` |
| Kondisi lahan dan posisi | `land_surface_condition`, `position`, `plan_configuration` |
| Material struktur | `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`, `superstructure` |
| Target | `damage_grade` (Grade 1–5, ordinal) |

`building_id` bernilai unik pada seluruh baris (533.465 nilai unik). Variabel `district_id`, `vdcmun_id`, dan `ward_id` bukan pengenal unik; ketiganya merupakan kode wilayah administratif sehingga diperlakukan sebagai kategorikal.

---

## 4. Hasil EDA dan Interpretasinya

### 4.1 Kualitas data

- **Nilai hilang:** 0 pada seluruh 17 kolom.
- **Duplikat:** 0 baris.
- **Anomali:** `age_building` memiliki nilai maksimum **999**, sedangkan persentil ke-75 hanya 27 tahun. Nilai 999 hampir pasti merupakan kode *placeholder* (misalnya "tidak diketahui"), bukan umur sebenarnya. Efeknya terlihat pada statistik: rata-rata umur 24,3 tahun dengan simpangan baku 65,0, jauh lebih besar dari rata-ratanya.

**Interpretasi:** "Tanpa nilai hilang" tidak berarti "tanpa masalah". Nilai sentinel seperti 999 secara diam-diam merusak statistik. Notebook pemodelan menanganinya dengan membuang 2.198 baris tersebut dari data latih (0,41%) dan mengecualikannya dari perhitungan rata-rata lingkungan.

### 4.2 Statistik deskriptif variabel numerik

| Variabel | Min | Median | Rata-rata | Persentil 75 | Maks |
|----------|----:|-------:|----------:|-------------:|-----:|
| `count_floors_pre_eq` | 1 | 2 | 2,09 | 2 | 9 |
| `age_building` (termasuk 999) | 0 | 16 | 24,34 | 27 | 999 |
| `plinth_area_sq_ft` | 70 | 358 | 406,52 | 470 | 5.000 |
| `height_ft_pre_eq` | 6 | 16 | 16,05 | 18 | 99 |

**Interpretasi:**

- Bangunan tipikal adalah rumah **dua lantai**, setinggi sekitar **16 kaki**, dengan luas dasar sekitar **358 kaki persegi** (±33 m²).
- Rata-rata melebihi median pada umur dan luas dasar, yang menandakan **distribusi menjulur ke kanan** (*right-skewed*) dengan ekor panjang bangunan tua dan besar.
- Nilai ekstrem seperti 9 lantai, tinggi 99 kaki, dan luas 5.000 kaki persegi sangat jauh dari kuartil ke-3. Nilai tersebut bisa jadi bangunan nyata yang jarang atau kesalahan entri. Notebook tidak menyelidikinya lebih jauh, dan model berbasis pohon relatif tahan terhadap pencilan sehingga tidak dilakukan pemangkasan.

### 4.3 Distribusi variabel kategorikal

Populasi bangunan sangat didominasi oleh konstruksi tradisional:

| Variabel | Kategori dominan | Proporsi |
|----------|------------------|---------:|
| `superstructure` | Mud mortar–stone | 77,9% |
| `foundation_type` | Mud mortar–stone/brick | 82,5% |
| `ground_floor_type` | Mud | 81,1% |
| `roof_type` | Bamboo/timber–light roof | 66,1% |
| `land_surface_condition` | Flat | 82,9% |
| `position` | Not attached | 79,3% |
| `plan_configuration` | Rectangular | 96,0% |

**Interpretasi:** Sebagian besar variabel kategorikal memiliki satu kategori yang sangat dominan. Konsekuensinya, kategori langka seperti bangunan RC berinsinyur (5.475 bangunan untuk `rc_engineered`), `H-shape` (61), dan `Building with Central Courtyard` (64) memiliki dukungan statistik yang kecil. Estimasi pengaruh kategori langka ini akan berderau tinggi. Hal ini menjelaskan keputusan di notebook pemodelan untuk menggabungkan bentuk denah selain Rectangular, Square, dan L-shape menjadi "Others", serta menggabungkan `Timber` ke `Other` pada `ground_floor_type`.

### 4.4 Distribusi target

| Kelas | Jumlah | Proporsi |
|-------|-------:|---------:|
| Grade 5 | 193.036 | 36,2% |
| Grade 4 | 128.691 | 24,1% |
| Grade 3 | 95.488 | 17,9% |
| Grade 2 | 61.080 | 11,4% |
| Grade 1 | 55.170 | 10,3% |

**Interpretasi:** Distribusi **miring ke arah kerusakan berat**: Grade 4 dan 5 bersama-sama mencakup 60,3% bangunan. Ini konsisten dengan dominasi konstruksi *mud mortar* yang rapuh terhadap guncangan. Ketidakseimbangan kelas (rasio kelas terbesar terhadap terkecil ≈ 3,5:1) menjelaskan mengapa akurasi saja tidak memadai dan **Macro F1** dipilih sebagai metrik utama.

### 4.5 Matriks asosiasi

Matriks asosiasi dihitung dengan pustaka `dython` yang memakai ukuran berbeda menurut tipe pasangan variabel (korelasi untuk numerik–numerik, rasio korelasi untuk kategorik–numerik, dan ukuran asosiasi nominal untuk kategorik–kategorik). Nilai berkisar 0 hingga 1 untuk ukuran nominal.

**Asosiasi dengan `damage_grade` (urut menurun):**

| Variabel | Asosiasi |
|----------|---------:|
| `ward_id` | 0,447 |
| `vdcmun_id` | 0,369 |
| `district_id` | 0,301 |
| `superstructure` | 0,300 |
| `roof_type` | 0,287 |
| `foundation_type` | 0,256 |
| `other_floor_type` | 0,250 |
| `ground_floor_type` | 0,225 |
| `count_floors_pre_eq` | 0,201 |
| `plinth_area_sq_ft` | 0,178 |
| `height_ft_pre_eq` | 0,066 |
| `age_building` | 0,060 |
| `plan_configuration` | 0,053 |
| `position` | 0,048 |
| `land_surface_condition` | 0,037 |

**Interpretasi:**

1. **Lokasi adalah penentu paling informatif.** Semakin halus resolusi wilayah (distrik → kecamatan → ward), semakin kuat asosiasinya (0,301 → 0,369 → 0,447). Ini sesuai dengan fisika gempa: intensitas guncangan, jenis tanah, dan jarak ke sumber gempa bervariasi secara spasial dan tidak tertangkap oleh atribut bangunan semata. Dua bangunan identik dapat mengalami kerusakan berbeda karena lokasinya.
2. **Material struktur lebih informatif daripada ukuran bangunan.** Atap, fondasi, dan lantai (0,22–0,29) berasosiasi lebih kuat dengan kerusakan daripada tinggi (0,066) dan umur (0,060). Ini selaras dengan prinsip teknik sipil bahwa kualitas material dan sistem struktural mengatur respons terhadap beban lateral.
3. **Umur dan tinggi berasosiasi lemah secara marginal.** Perlu hati-hati: nilai rendah pada ukuran asosiasi marginal tidak membuktikan bahwa variabel tersebut tidak penting. Pengaruhnya bisa bersifat interaksi (misalnya, bangunan tinggi dan berbahan lumpur), yang tidak tertangkap oleh asosiasi berpasangan. Karena itu notebook pemodelan membuat fitur interaksi dan rasio.
4. **Kondisi lahan dan posisi hampir tidak berasosiasi** secara marginal (0,037 dan 0,048). Hasil ini sedikit berlawanan dengan ekspektasi teknik bahwa lereng curam memperburuk kerusakan. Penjelasan yang mungkin adalah kategori "Steep slope" hanya 3,3% data (17.432 bangunan), sehingga sinyalnya tertutup. Ini hipotesis, bukan temuan terbukti.
5. **Redundansi antarfitur tinggi:**
   - `district_id`, `vdcmun_id`, `ward_id` saling berasosiasi ≈ 0,999 karena bersarang secara hierarkis (satu ward hanya milik satu kecamatan dan satu distrik). Ketiganya membawa informasi yang hampir sama pada resolusi berbeda.
   - `count_floors_pre_eq` dengan `other_floor_type` (0,723) dan dengan `height_ft_pre_eq` (0,712). Lebih banyak lantai tentu berarti lebih tinggi dan perlu tipe lantai atas.
   - `foundation_type` dengan `superstructure` (0,583) dan `roof_type` (0,573): material cenderung dipakai sepaket.

   Model pohon tidak terlalu terpengaruh multikolinearitas, tetapi redundansi ini membuat interpretasi pentingnya fitur individual menjadi ambigu.

**Peringatan metodologis:** ukuran asosiasi untuk variabel berkardinalitas tinggi seperti `ward_id` (945 level) cenderung menggelembung secara mekanis, karena banyak level memberi kebebasan lebih untuk "menjelaskan" target. Nilai 0,447 untuk `ward_id` sebaiknya dibaca sebagai batas atas sinyal lokasi, bukan estimasi tak bias.

---

## 5. Metodologi Pemodelan

### 5.1 Pembagian data

- Data latih bersih (531.267 baris) dibagi **80:20** secara **stratified** berdasarkan `damage_grade`, `random_state=42`.
- Hasilnya: 425.013 baris latih dan **106.254 baris validasi**.
- Pembagian bersifat **acak**, bukan spasial atau temporal (lihat Bagian 7.2).

### 5.2 Rekayasa fitur (`MicroStreetPreprocessor`, total 55 fitur)

| Kelompok | Contoh fitur | Rasional |
|----------|--------------|----------|
| **Geometri teknik sipil** | `height_area_ratio`, `slenderness_ratio`, `building_volume`, `avg_floor_height`, `area_per_floor`, `total_floor_area`, `is_very_old` (>50 th), `age_per_floor` | Kelangsingan dan massa memengaruhi respons dinamis terhadap gempa. |
| **Interaksi material** | `foundation_superstructure`, `roof_floor_combo`, `surface_position_combo` | Pengaruh material bersifat kombinatorial, bukan aditif. |
| **Dekode `building_id`** | `old_vdc_id`, `old_ward_id`, `street_block_id`, `survey_seq` | ID mengandung hierarki wilayah dan urutan survei. |
| **Kepadatan wilayah** | `ward_building_count`, `old_ward_building_count`, `vdcmun_building_count`, dll. | Proksi kepadatan permukiman. |
| **Statistik relatif wilayah** | `height_vs_old_ward`, `age_vs_old_ward`, `area_vs_old_ward` | Apakah bangunan lebih tinggi/tua/luas dibanding sekitarnya. |
| **Tetangga jalan (urutan survei)** | `seq_diff_prev/next`, `survey_seq_rel`, `street5_mean_*`, `same_super_neighbors`, `same_found_neighbors`, `height_vs_street5`, dll. | Bangunan bernomor urut berdekatan di *old ward* yang sama diasumsikan bertetangga fisik. |

Statistik wilayah dan tetangga dihitung dari **seluruh sensus** (latih + validasi + uji, sekitar 760.000 bangunan) **tanpa memakai label**. Ini adalah pendekatan transduktif yang sah selama tidak menyentuh target.

### 5.3 Model

| Model | Konfigurasi utama | Peran |
|-------|-------------------|-------|
| **LightGBM** | 1.200 pohon, `learning_rate=0,03`, `num_leaves=85`, `max_depth=16`, `max_bin=1023`, `colsample=0,75`, `subsample=0,8` | Model gradient boosting utama |
| **XGBoost** | 1.100 pohon, `learning_rate=0,03`, `max_depth=9`, `max_bin=512`, kategorikal native, `hist` | Pembanding gradient boosting |
| **StreetNeighborKNN** | `cKDTree` 1-D, `k_street=20`, `k_struct=15`; 45% tetangga jalan + 40% tetangga sematerial + 15% prior ward | Model spesialis yang memanfaatkan label tetangga berdasarkan urutan survei |

### 5.4 Ensemble dan penyesuaian ambang keputusan

1. Probabilitas ketiga model digabung linear: `p = w_lgb·p_lgb + w_xgb·p_xgb + w_knn·p_knn`.
2. Bobot `w_knn ∈ {0; 0,08; 0,12; 0,16; 0,22}` dan rasio LGBM:XGB ∈ {0,45; 0,50; 0,55} dicari secara *grid*.
3. Untuk tiap kombinasi, **bobot kelas** (vektor 5 elemen pengali probabilitas sebelum `argmax`) dioptimalkan dengan metode Powell untuk memaksimalkan Macro F1.
4. Konfigurasi terbaik dikunci, lalu model dilatih ulang pada **100% data latih bersih** untuk prediksi data uji kompetisi.

---

## 6. Hasil Pemodelan dan Interpretasinya

### 6.1 Skor model individual (tanpa tuning, pada 106.254 sampel validasi)

| Model | Macro F1 |
|-------|---------:|
| LightGBM | **0,58405** |
| XGBoost | 0,57971 |
| StreetNeighborKNN | 0,55831 |

**Interpretasi:**

- Kedua model *boosting* hampir setara (selisih 0,0043). Perbedaan sekecil ini kemungkinan masih berada dalam rentang variasi akibat *seed* dan pembagian data; notebook tidak memberi interval kepercayaan sehingga superioritas LightGBM tidak dapat dinyatakan secara statistik.
- **KNN spasial sederhana yang hanya memakai label tetangga dan prior wilayah mencapai 0,558**, hanya sekitar 2,6 poin persentase di bawah LightGBM dengan 55 fitur. Ini temuan yang paling informatif: **sebagian besar sinyal prediktif berada pada kedekatan urutan survei**, yaitu bangunan yang disurvei berurutan cenderung mengalami kerusakan serupa. Hal ini sesuai dengan autokorelasi spasial kerusakan gempa, sekaligus menandakan bahwa struktur `building_id` sangat berperan.

### 6.2 Ensemble

| Parameter | Nilai |
|-----------|------:|
| Proporsi LightGBM | 0,429 |
| Proporsi XGBoost | 0,351 |
| Proporsi StreetKNN | 0,220 |
| Bobot kelas (G1, G2, G3, G4, G5) | 0,7933; 0,9358; 0,9130; 0,7600; 0,5338 |
| **Macro F1 (validasi)** | **0,59402** |
| Rekor lokal sebelumnya | 0,56140 |

Peningkatan: +0,0100 terhadap LightGBM tunggal; +0,0326 terhadap rekor lokal sebelumnya; +0,0357 terhadap KNN tunggal.

**Interpretasi:**

- **Penggabungan memberi peningkatan kecil tapi konsisten** (≈1 poin persentase). Ensemble efektif ketika model membuat kesalahan yang tidak sepenuhnya berkorelasi. KNN spasial berbasis label tetangga dan model pohon berbasis fitur memang memanfaatkan sumber sinyal yang berbeda, sehingga masuk akal bahwa KNN memperoleh porsi 22%.
- **Bobot KNN optimal (0,22) berada di batas atas grid** yang dicari. Artinya, optimum sebenarnya mungkin lebih tinggi dan grid belum memuat titik jenuh.
- **Bobot kelas** menekan Grade 5 paling kuat (0,53) dan Grade 1 serta Grade 4 secara moderat (0,79 dan 0,76), sedangkan Grade 2 dan 3 nyaris tidak ditekan (0,94 dan 0,91). Ini memperlihatkan mekanisme optimasi Macro F1: mengurangi dominasi kelas mayoritas agar kelas tengah yang sulit ikut terprediksi. Nilai Grade 4 (0,7600) identik dengan titik awal Powell, menandakan bahwa pengoptimal tidak bergerak pada dimensi tersebut. Ini bisa berarti permukaan optimasi datar di sana atau terhenti di optimum lokal.

### 6.3 Laporan klasifikasi per kelas (ensemble, validasi)

| Kelas | Presisi | Recall | F1 | Support |
|-------|--------:|-------:|---:|--------:|
| Grade 1 | 0,67 | 0,72 | 0,70 | 10.982 |
| Grade 2 | 0,46 | 0,46 | 0,46 | 12.166 |
| Grade 3 | 0,46 | 0,52 | 0,49 | 19.011 |
| Grade 4 | 0,55 | 0,58 | 0,57 | 25.643 |
| Grade 5 | 0,80 | 0,71 | 0,75 | 38.452 |
| **Akurasi** | | | **0,62** | 106.254 |
| **Macro avg** | 0,59 | 0,60 | **0,59** | |
| **Weighted avg** | 0,63 | 0,62 | 0,62 | |

**Interpretasi:**

1. **Kelas ekstrem lebih mudah dibedakan.** Grade 5 (kerusakan total, F1 0,75) dan Grade 1 (kerusakan minimal, F1 0,70) adalah kondisi yang paling berbeda secara fisik sehingga paling mudah dikenali.
2. **Kelas menengah (Grade 2 dan 3) adalah titik lemah** (F1 0,46 dan 0,49). Presisi dan recall Grade 2 sama-sama 0,46, artinya hampir separuh prediksi Grade 2 salah dan separuh bangunan Grade 2 tidak terdeteksi. Pola ini lazim pada target **ordinal**: batas antara tingkat kerusakan berdekatan bersifat kabur, dan penilaian lapangan antar-surveyor pun berpotensi subjektif. Notebook tidak menyertakan matriks konfusi, sehingga arah kesalahan (misalnya Grade 2 ↔ Grade 3) belum dikonfirmasi, tetapi pola F1 mendukung dugaan tersebut.
3. **Grade 5 memiliki presisi tinggi (0,80) tetapi recall lebih rendah (0,71).** Ini konsekuensi langsung dari bobot kelas 0,53 yang sengaja "mengerem" prediksi Grade 5 demi keseimbangan Macro F1 (akibatnya sebagian Grade 5 diprediksi sebagai Grade 4 atau 3).
4. **Akurasi 0,62** berarti sekitar 38% bangunan salah diklasifikasi. Untuk masalah lima kelas dengan kelas tengah yang kabur, ini adalah performa moderat dan bukan hasil yang siap dipakai sebagai satu-satunya dasar keputusan lapangan.

### 6.4 Prediksi pada data uji kompetisi

Model akhir dilatih pada 531.267 baris dan menghasilkan 228.629 prediksi (tanpa nilai kosong).

| Kelas | Prediksi uji | Proporsi prediksi | Proporsi data latih |
|-------|-------------:|------------------:|--------------------:|
| Grade 5 | 72.330 | 31,6% | 36,2% |
| Grade 4 | 57.668 | 25,2% | 24,1% |
| Grade 3 | 46.816 | 20,5% | 17,9% |
| Grade 2 | 26.548 | 11,6% | 11,4% |
| Grade 1 | 25.267 | 11,1% | 10,3% |

**Interpretasi:** Distribusi prediksi lebih **merata** daripada distribusi latih: Grade 5 berkurang 4,6 poin persentase, sedangkan Grade 3 bertambah 2,6 poin persentase. Ini konsisten dengan efek bobot kelas yang disengaja. Perlu dicatat bahwa pergeseran ini tidak dengan sendirinya membuktikan kebenaran, karena distribusi sebenarnya pada data uji tidak diketahui. Notebook juga tidak melaporkan skor leaderboard, sehingga performa pada data uji tersembunyi belum dapat dinilai dari bukti yang ada.

---

## 7. Diskusi: Validitas dan Keterbatasan

### 7.1 Optimisme pada skor validasi (*tuning on the evaluation set*)

Bobot ensemble dan bobot kelas dioptimalkan **langsung pada himpunan validasi** (`y_test_num`), lalu skor Macro F1 dilaporkan dari himpunan yang sama. Prosedur ini menyebabkan estimasi bias ke atas karena parameter ikut "menyesuaikan" pada derau data validasi. Dengan 106.254 sampel dan hanya 8 parameter bebas (3 proporsi model dan 5 bobot kelas), bias kemungkinan kecil, tetapi besarnya tidak terukur dalam notebook. Skor yang lebih jujur adalah **0,584** (LightGBM tanpa tuning). Peningkatan sebesar 0,010 oleh ensemble dan tuning sebaiknya diverifikasi pada *hold-out* terpisah atau validasi silang bersarang (*nested*).

### 7.2 Kebocoran spasial dan ketergantungan pada `building_id`

Ini keterbatasan paling penting.

- Pembagian latih/validasi bersifat **acak**. Bangunan validasi sering bertetangga dengan bangunan latih pada *old ward* yang sama, bahkan bernomor urut berdekatan. Model KNN secara eksplisit menyalin label tetangga tersebut. Validasi acak dengan demikian **menaksir terlalu tinggi** kemampuan model pada wilayah yang belum pernah disurvei.
- Fitur `survey_seq`, `old_ward_id`, dan `street_block_id` diturunkan dari *building_id*, yaitu pengenal administratif, bukan atribut fisik bangunan. Fitur ini valid dalam konteks kompetisi karena data uji berasal dari sensus yang sama dengan format ID yang sama, tetapi **tidak akan tersedia atau bermakna pada gempa lain, wilayah lain, atau bangunan baru tanpa skema ID serupa**.
- Dugaan bahwa fitur tetangga jalan adalah sumber utama peningkatan dari 0,5614 ke 0,594 adalah *hipotesis yang masuk akal*, tetapi **tidak dibuktikan** karena notebook tidak memuat studi ablasi maupun *feature importance*.

**Implikasi:** model ini cocok sebagai alat bantu *dalam sensus yang sedang berlangsung* (ketika sebagian bangunan sudah dinilai dan sisanya diprediksi), tetapi belum teruji untuk penggunaan umum.

### 7.3 Target ordinal diperlakukan nominal

`damage_grade` diurutkan secara alami (Grade 1 < … < Grade 5), tetapi model diperlakukan sebagai klasifikasi multikelas biasa dan dievaluasi dengan Macro F1 yang tidak membedakan kesalahan "dekat" (Grade 3 → 4) dari kesalahan "jauh" (Grade 1 → 5). Metrik sensitif-ordinal (misalnya *quadratic weighted kappa* atau MAE ordinal) dan model ordinal belum dicoba.

### 7.4 Keterbatasan lain

- **Tidak ada interval kepercayaan atau pengulangan dengan *seed* berbeda**, sehingga selisih kecil antarmodel (misalnya LightGBM vs XGBoost) tidak dapat disimpulkan signifikan.
- **Tidak ada matriks konfusi, kalibrasi probabilitas, maupun analisis kesalahan** per wilayah atau material.
- **Tidak ada *feature importance* atau SHAP**, sehingga interpretasi "fitur apa yang penting" pada laporan ini bersandar pada EDA marginal, bukan pada model akhir.
- **Hiperparameter tidak disetel secara sistematis** (nilai tampak ditetapkan manual); potensi peningkatan dari *tuning* belum dieksplorasi.
- **Penanganan umur:** 2.198 baris berumur 999 dibuang dari data latih, tetapi bangunan serupa pada data uji tetap harus diprediksi. Notebook tidak menunjukkan penanganan khusus bagi kasus ini pada data uji, selain menjaga nilai 999 dari rata-rata lingkungan.
- **Generalisasi kausal:** seluruh hubungan dalam laporan ini bersifat asosiatif. Tidak ada klaim bahwa material tertentu *menyebabkan* kerusakan.

---

## 8. Kesimpulan

1. **Data** berukuran besar (533 ribu bangunan), lengkap, tanpa duplikat, dengan satu anomali nyata (`age_building = 999`) yang telah ditangani.
2. **Kerusakan berat mendominasi** (Grade 4–5 = 60,3%), dan stok bangunan didominasi konstruksi tradisional *mud mortar*, sesuai dengan kerentanan terhadap guncangan.
3. **Lokasi dan material struktur** adalah kelompok variabel yang paling berasosiasi dengan kerusakan; dimensi fisik (tinggi, umur) dan kondisi lahan berasosiasi lebih lemah secara marginal.
4. **Ensemble tiga model** mencapai **Macro F1 ≈ 0,594** (akurasi 0,62). Dengan koreksi untuk bias tuning, estimasi yang lebih hati-hati berada di sekitar **0,58–0,59**.
5. **Kelas tengah (Grade 2 dan 3) merupakan hambatan utama** akurasi, sedangkan kelas ekstrem relatif mudah dikenali.
6. **Sinyal terkuat bersifat spasial-administratif**: KNN sederhana berbasis urutan survei sudah mendekati performa model berfitur penuh. Ini kekuatan dalam konteks kompetisi, tetapi sekaligus batas generalisasi model.

---

## 9. Rekomendasi Tindak Lanjut

Urutan berikut disusun menurut dampak terhadap kesahihan kesimpulan:

1. **Pisahkan himpunan penyetelan dan evaluasi.** Optimalkan bobot ensemble dan bobot kelas dengan validasi silang pada data latih, lalu laporkan skor pada *hold-out* yang tidak dipakai untuk tuning.
2. **Validasi spasial.** Gunakan *GroupKFold* berdasarkan `ward_id` atau `old_ward_id` agar estimasi mencerminkan prediksi pada wilayah baru, lalu bandingkan dengan validasi acak untuk mengukur besar kebocoran spasial.
3. **Studi ablasi.** Latih model tanpa fitur turunan `building_id` dan tanpa KNN, untuk memisahkan kontribusi fitur fisik murni dari sinyal urutan survei.
4. **Tambahkan interpretabilitas:** *feature importance*, SHAP, dan matriks konfusi untuk mengonfirmasi pola kesalahan Grade 2 ↔ 3.
5. **Perlakukan target sebagai ordinal:** coba regresi ordinal, pendekatan *cumulative link*, atau optimasi ambang pada skor kontinu, serta laporkan metrik sensitif-ordinal.
6. **Perluas grid bobot KNN** melampaui 0,22 karena optimum berada di tepi grid.
7. **Kuantifikasi ketidakpastian:** ulangi pelatihan dengan beberapa *seed* dan laporkan rata-rata ± simpangan baku, serta interval kepercayaan *bootstrap* untuk Macro F1.
8. **Selidiki pencilan** (tinggi 99 kaki, 9 lantai, luas 5.000 kaki persegi) dan makna kode 999 bersama pemilik data.
9. **Kalibrasi probabilitas** bila keluaran akan dipakai untuk pengambilan keputusan berbasis risiko.

---

## 10. Struktur Repositori dan Reproduksibilitas

```
.
├── data/
│   └── train.csv                # dipakai di 01_eda.ipynb (../data/train.csv)
├── notebooks/
│   ├── 01_eda.ipynb             # EDA: kualitas data, distribusi, matriks asosiasi
│   └── 02_modeling.ipynb        # Fitur, model, ensemble, evaluasi, submission
└── README.md
```

**Lingkungan.** Notebook modeling dijalankan di Kaggle dengan data kompetisi `structural-damage-classification` (`train.csv`, `test.csv`, `metadata.csv`, `sample_submission.csv`).

**Pustaka utama:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `category_encoders`, `scipy` (`cKDTree`, `optimize.minimize`), `dython` (matriks asosiasi), `matplotlib`, `seaborn`.

**Catatan:** struktur folder di atas adalah asumsi berdasarkan path `../data/train.csv` pada notebook EDA; sesuaikan dengan repositori Anda.

**Urutan eksekusi:**

1. Jalankan `01_eda.ipynb` untuk eksplorasi (memerlukan `../data/train.csv`).
2. Jalankan `02_modeling.ipynb` secara berurutan. Sel 7 menjalankan validasi lokal dan pencarian bobot; sel 8 melatih ulang pada 100% data dan menulis `submission9_ultimate_microstreet_100pct.csv`.

**Reproduksibilitas:** `random_state=42` dipakai pada pembagian data dan seluruh model. Hasil dapat bergeser sedikit akibat paralelisme (`n_jobs=-1`) dan versi pustaka. Tahap pelatihan memerlukan RAM besar, dan notebook sengaja melepas objek dengan `gc.collect()` di antara tahap.

---

*Seluruh angka bersumber dari keluaran notebook yang dilampirkan. Pernyataan bertanda "hipotesis", "kemungkinan", atau "dugaan" adalah interpretasi penulis yang belum diuji oleh eksperimen dalam notebook.*
