# Greenhouse Anomaly Detection Edge AI

Repositori ini berisi proyek Edge Machine Learning untuk deteksi anomali pada data iklim indoor greenhouse menggunakan model Autoencoder, dan disiapkan untuk deployment di ESP32.

Alur kerjanya:

1. Unduh repositori dari GitHub
2. Ekstrak file ZIP di komputer lokal
3. Unggah folder `greenhouse-anomaly-edgeai` ke Google Drive
4. Jalankan tiga file Jupyter Notebook untuk feature analysis, training model, dan konversi TensorFlow Lite
5. Periksa file model yang dihasilkan di folder `models`
6. Siapkan lingkungan deployment ESP32 di komputer lokal
7. Instal Arduino IDE, board package ESP32, driver USB atau serial, dan `Chirale_TensorFlowLite`
8. Buat sketch Arduino yang memakai `greenhouse_ae.h`
9. Pilih board ESP32 dan port COM yang benar
10. Compile dan upload program ke ESP32
11. Buka Serial Monitor
12. Kirim satu window data greenhouse, lalu baca MSE, threshold, dan hasil akhir `NORMAL` atau `ANOMALY`

## Dataset

**Dindigul Greenhouse Indoor Dataset (2024 dan 2025)**

Dataset tersedia di direktori `datasets`. Isinya pengukuran iklim indoor greenhouse per jam dari Dindigul.

- `dindigul_greenhouse_indoor_2024_dirty.csv` dipakai sebagai data latih (8783 baris).
- `dindigul_greenhouse_indoor_2025_dirty.csv` dipakai sebagai data evaluasi (8759 baris).

Kolom pada file mentah:

```text
datetime, indoor_temp, indoor_humidity, indoor_air_velocity, indoor_CO2,
solarradiation, day_night_flag, vpd, dew_point, leaf_wetness_proxy
```

File mentah ini masih kotor. Ada nilai kosong, timestamp ganda, jam yang hilang, nilai mustahil (misalnya `-999` pada CO2), dan satu kolom yang sebagian besar kosong (`indoor_air_velocity`). Notebook 1 membersihkannya dan menyimpan hasilnya di `datasets/clean/`.

Dataset tidak punya label anomali. Anomali sintetis disuntikkan ke window normal untuk mengevaluasi model.

## Struktur Repositori

```text
greenhouse-anomaly-edgeai/
├── README.md
├── notebooks/
│   ├── greenhouse_anomaly_feature_analysis.ipynb
│   ├── greenhouse_anomaly_training_autoencoder.ipynb
│   ├── greenhouse_anomaly_tflite_conversion.ipynb
│   └── c_writer.py
├── datasets/
│   ├── dindigul_greenhouse_indoor_2024_dirty.csv
│   ├── dindigul_greenhouse_indoor_2025_dirty.csv
│   └── clean/
│       ├── dindigul_greenhouse_indoor_2024_clean.csv
│       ├── dindigul_greenhouse_indoor_2025_clean.csv
│       ├── dindigul_greenhouse_indoor_2024_imputed_mask.csv
│       ├── dindigul_greenhouse_indoor_2025_imputed_mask.csv
│       └── feature_analysis_config.json
└── models/
    ├── greenhouse_ae_model_ae.h5
    ├── greenhouse_ae_model_ae_float32.tflite
    ├── greenhouse_ae_model_ae_float16.tflite
    ├── greenhouse_ae_model_ae_dynamic.tflite
    ├── greenhouse_ae_model_ae_int8.tflite
    ├── greenhouse_ae_model_dae.h5
    ├── greenhouse_ae_model_dae_float32.tflite
    ├── greenhouse_ae_model_dae_float16.tflite
    ├── greenhouse_ae_model_dae_dynamic.tflite
    ├── greenhouse_ae_model_dae_int8.tflite
    ├── greenhouse_scaler.joblib
    ├── greenhouse_config.json
    ├── greenhouse_normal_anomaly_samples.npz
    └── esp32_deploy_tflite/
        └── greenhouse_ae.h
```

File model dengan akhiran `_ae` adalah Dense Autoencoder. File dengan akhiran `_dae` adalah Denoising Dense Autoencoder.

## Bagian 1: Unduh Repositori dari GitHub

1. Buka repositori GitHub.
2. Klik tombol hijau **Code**.
3. Pilih **Download ZIP**.
4. Tunggu sampai file ZIP selesai diunduh.
5. Ekstrak file ZIP di komputer lokal.

Setelah diekstrak, kamu akan mendapat folder `greenhouse-anomaly-edgeai` yang berisi folder `notebooks`, `datasets`, dan `models`.

## Bagian 2: Unggah Folder Proyek ke Google Drive

Ketiga notebook adalah file Jupyter Notebook (`.ipynb`) yang ditulis untuk Google Colab. Notebook membaca dan menulis file lewat path berikut:

```text
/content/drive/MyDrive/greenhouse-anomaly-edgeai
```

1. Buka **Google Drive**.
2. Buka **My Drive**.
3. Unggah seluruh folder `greenhouse-anomaly-edgeai` (bukan hanya notebook).
4. Pastikan folder berada langsung di **My Drive**, supaya path di atas valid.
5. Buka folder `notebooks`. Kamu akan menemukan tiga file berikut:

   - `greenhouse_anomaly_feature_analysis.ipynb`
   - `greenhouse_anomaly_training_autoencoder.ipynb`
   - `greenhouse_anomaly_tflite_conversion.ipynb`

6. Buka notebook yang ingin dijalankan.
7. Jika Google Drive membuka file di viewer, pilih **Open with Google Colaboratory** bila tersedia.
8. Jalankan sel-sel awal setiap notebook untuk me-mount Google Drive dan izinkan akses saat diminta.

Jika lokasi Drive berbeda, ubah `BASE_PATH` pada sel pengaturan di setiap notebook.

Jalankan notebook secara berurutan, karena tiap notebook memakai keluaran notebook sebelumnya.

### Notebook 1: Feature Analysis

Buka:

```text
greenhouse_anomaly_feature_analysis.ipynb
```

Notebook ini mengeksplorasi data kotor, mencari masalah pada data, membersihkan data, dan menganalisis fitur yang bisa memisahkan window normal dari window anomali.

Langkah utama:

1. Kenali data mentah: ukuran, missing value, statistik, duplikat
2. Periksa interval waktu, nilai mustahil, outlier, skewness, korelasi, dan pergeseran antar tahun
3. Investigasi masalah data: nilai mustahil, missing value, kandidat sensor macet, pola CO2, `day_night_flag`, RH jenuh
4. Bersihkan data: buang `indoor_air_velocity`, pilih metode imputasi dengan masked benchmark, hitung ulang `vpd`
5. Validasi data bersih lalu simpan
6. Buat window dan pembagian train, validation, test berbasis blok untuk mencegah kebocoran data
7. Suntikkan anomali sintetis dan ekstrak fitur statistik per window
8. Ranking fitur (ROC-AUC), jalankan PCA dan FFT, lalu lakukan scaling, encoding, dan binning

Sebelum dijalankan, pastikan `BASE_PATH` mengarah ke folder proyek yang sudah diunggah.

Keluaran:

```text
datasets/clean/
```

### Notebook 2: Training Autoencoder

Buka:

```text
greenhouse_anomaly_training_autoencoder.ipynb
```

Notebook ini melatih Autoencoder memakai data bersih dari Notebook 1. Training hanya memakai window normal. Satu window mencakup 24 jam dengan hop 1 jam dan 8 fitur.

Atur `model_type` pada sel pengaturan:

```text
ae   Dense Autoencoder (default, dipakai oleh header deployment yang disediakan)
dae  Denoising Dense Autoencoder
vae  Variational Autoencoder
```

Langkah utama:

1. Muat data bersih dan bagi menjadi blok train, validation, dan test
2. Fit `StandardScaler` hanya pada data latih
3. Susun set evaluasi: window 2025 yang belum pernah dilihat dan anomali sintetis
4. Bangun dan latih Autoencoder dengan early stopping
5. Hitung MSE per window dan tentukan threshold anomali
6. Evaluasi dengan confusion matrix, precision, recall, F1, dan ROC-AUC
7. Periksa recall per skenario dan peta error per jam dan per fitur
8. Simpan model, scaler, config, dan sampel

Model hasil training disimpan sebagai model Keras `.h5`. Notebook juga menyimpan:

```text
models/greenhouse_ae_model_<model_type>.h5
models/greenhouse_scaler.joblib
models/greenhouse_config.json
models/greenhouse_normal_anomaly_samples.npz
```

Repositori ini sudah berisi file siap pakai di `models/`, jadi training boleh dilewati jika kamu hanya ingin mengonversi atau men-deploy model yang ada.

### Notebook 3: Konversi TensorFlow Lite

Buka:

```text
greenhouse_anomaly_tflite_conversion.ipynb
```

Notebook ini mengonversi model Keras hasil training ke format TensorFlow Lite dan menyiapkan model untuk mikrokontroler.

Langkah utama:

1. Muat model `.h5`, scaler, sampel, dan `greenhouse_config.json`
2. Siapkan 300 window normal sebagai representative dataset untuk kuantisasi int8
3. Konversi ke TFLite: `float32`, `dynamic`, `float16`, dan `int8`
4. Bandingkan ukuran file
5. Jalankan TFLite interpreter dan bandingkan dengan Keras (AUC dan agreement)
6. Uji satu window normal dan satu window anomali untuk mendapat nilai known-good
7. Tulis model int8 sebagai header C

File deployment ditulis ke:

```text
models/esp32_deploy_tflite/greenhouse_ae.h
```

Notebook menulis `c_writer.py` ke folder `notebooks` secara otomatis. Kamu tidak perlu membuatnya manual.

## Ringkasan Model

Nilai di bawah dibaca dari `models/greenhouse_config.json`.

| Item | Nilai |
|---|---|
| Tipe model | Dense Autoencoder (Flatten, Dense, Reshape) |
| Window | 24 jam, hop 1 jam |
| Fitur (8) | `indoor_temp`, `indoor_humidity`, `indoor_CO2`, `solarradiation`, `vpd`, `dew_point`, `leaf_wetness_proxy`, `day_night_flag` |
| Ukuran input | 24 x 8 = 192 nilai |
| Hidden units | 128, 64 |
| Bottleneck | 32 |
| Fitur yang dibuang | `indoor_air_velocity` (sekitar 45 persen kosong) |
| Threshold anomali | 0.097878597676754 (MSE pada data ter-scale) |
| Model deployment | `greenhouse_ae_model_ae_int8.tflite` (92224 byte) |

Siang didefinisikan sebagai `solarradiation > 1.0`. Kolom `day_night_flag` tidak dipakai untuk menentukan siang karena bergeser sekitar 7 jam dari matahari.

## Bagian 3: Siapkan Lingkungan Deployment ESP32 di Komputer Lokal

Sebelum men-deploy model ke ESP32, siapkan komputer lokal dengan tools pengembangan, dukungan board ESP32, driver USB atau serial, dan library TensorFlow Lite.

### Instal Arduino IDE

1. Unduh dan instal **Arduino IDE** di komputer lokal.
2. Buka Arduino IDE setelah instalasi selesai.
3. Pastikan Arduino IDE bisa berjalan normal sebelum lanjut.

### Instal Board Package ESP32

1. Buka Arduino IDE.
2. Instal board package ESP32 lewat **Board Manager** Arduino IDE.
3. Pastikan board ESP32 tersedia di:

```text
Tools > Board
```

### Instal Driver USB atau Serial ESP32

1. Hubungkan ESP32 ke komputer dengan kabel USB.
2. Jika ESP32 tidak muncul sebagai serial port, instal driver USB-to-Serial yang sesuai dengan chip USB pada board ESP32 kamu.

Chip USB-to-Serial yang umum:

```text
CP210x
CH340
```

Lihat chip kecil di dekat port USB pada board ESP32 untuk mengetahui chip yang dipakai (biasanya bertanda `CP2102` atau `CH340`).

Link unduhan driver resmi:

- **CP210x (Silicon Labs)** untuk Windows, macOS, Linux, dan Android:
  https://www.silabs.com/software-and-tools/usb-to-uart-bridge-vcp-drivers?tab=downloads
- **CH340 / CH341 (WCH)** untuk Windows:
  https://www.wch.cn/download/CH341SER_EXE.html
- **CH340 / CH341 (WCH)** untuk Windows (versi ZIP):
  https://www.wch.cn/download/CH341SER_ZIP.html
- **CH340 / CH341 (WCH)** untuk Linux (opsional, kernel Linux biasanya sudah punya driver bawaan):
  https://github.com/WCHSoftGroup/ch341ser_linux

Jika ESP32 sudah muncul di `Tools > Port` tanpa instalasi apa pun, berarti driver sudah tersedia.

Setelah driver terpasang:

1. Cabut ESP32.
2. Hubungkan kembali ESP32 ke komputer.
3. Buka Arduino IDE.
4. Periksa:

```text
Tools > Port
```

ESP32 harus muncul sebagai serial port atau port COM yang tersedia.

> **Catatan:** Nomor port COM tergantung komputer dan bisa berbeda di komputer lain.

### Instal Library TensorFlow Lite

Proyek ini memakai library `Chirale_TensorFlowLite` untuk inferensi TensorFlow Lite Micro di ESP32.

Instal:

```text
Chirale_TensorFlowLite
```

Pastikan library sudah terpasang dan dikenali Arduino IDE sebelum compile. Jika Arduino IDE melaporkan header TensorFlow Lite tidak ditemukan saat compile, periksa kembali instalasi library tersebut.

## Bagian 4: Siapkan File Deployment ESP32

Folder deployment berisi model dalam bentuk header C:

```text
models/esp32_deploy_tflite/greenhouse_ae.h
```

Header ini mendefinisikan:

```text
greenhouse_ae       model TFLite int8 dalam bentuk array byte
greenhouse_ae_len   ukuran model dalam byte (92224)
```

Folder ini tidak berisi sketch Arduino (`.ino`). Buat sendiri di Arduino IDE, lalu letakkan di folder yang sama dengan header:

1. Di Arduino IDE, buat sketch baru dan simpan dengan nama `esp32_deploy_tflite`.
2. Salin `greenhouse_ae.h` ke folder sketch.
3. Tambahkan `#include "greenhouse_ae.h"` di sketch.

Sketch harus mengikuti preprocessing yang sama dengan notebook, kalau tidak hasilnya tidak valid:

1. Ambil satu window 24 jam dengan 8 fitur, urutannya sama dengan daftar `features` di `greenhouse_config.json`.
2. Scale nilai memakai mean dan scale yang tersimpan di `models/greenhouse_scaler.joblib`.
3. Kuantisasi nilai ter-scale ke int8 memakai scale dan zero point input model.
4. Jalankan inferensi dengan TensorFlow Lite Micro. Daftarkan operator yang dipakai model.
5. Dekuantisasi keluaran untuk mendapat window hasil rekonstruksi.
6. Hitung MSE antara input ter-scale dan hasil rekonstruksi.
7. Bandingkan MSE dengan threshold `0.097878597676754`.

Gunakan sampel normal dan anomali known-good yang ditampilkan Notebook 3 (bagian 10) untuk memastikan hasil di ESP32 sama dengan hasil di Python.

## Bagian 5: Konfigurasi Arduino IDE untuk ESP32

Sebelum compile sketch:

1. Buka sketch di Arduino IDE.
2. Hubungkan ESP32 ke komputer dengan kabel USB.
3. Pilih board ESP32 yang sesuai dengan hardware kamu:

```text
Tools > Board
```

4. Pilih port COM yang terhubung ke ESP32:

```text
Tools > Port
```

5. Pastikan board package ESP32 dan library `Chirale_TensorFlowLite` sudah terpasang.

## Bagian 6: Compile dan Upload

Setelah memilih board dan port:

1. Klik **Verify/Compile**.
2. Tunggu sampai compile selesai.
3. Jika tidak ada error, klik **Upload**.
4. Tunggu sampai Arduino IDE menyatakan upload selesai.
5. Biarkan ESP32 tetap terhubung ke komputer.

Jika compile gagal karena library tidak ditemukan, pastikan `Chirale_TensorFlowLite` terpasang dengan benar lalu compile ulang.

Jika ESP32 tidak terdeteksi, periksa kabel USB, driver USB atau serial, dan port COM yang dipilih.

## Bagian 7: Buka Serial Monitor

Setelah program di-upload:

1. Buka:

```text
Tools > Serial Monitor
```

2. Atur baud rate sesuai yang dipakai di sketch (misalnya `115200`).
3. Tekan tombol reset ESP32 jika perlu.
4. Tunggu pesan inisialisasi.

Startup yang berhasil seharusnya melaporkan bahwa model termuat, interpreter dibuat, dan tensor teralokasi. Pesan persisnya tergantung sketch kamu.

## Bagian 8: Jalankan Inferensi dan Baca Hasilnya

Model membutuhkan satu window penuh (192 nilai), bukan satu pembacaan tunggal. Kirim window lewat Serial Monitor dengan format yang diharapkan sketch kamu, atau simpan satu window uji di dalam sketch.

ESP32 akan:

1. Membaca window
2. Melakukan scaling dan kuantisasi input
3. Menjalankan inferensi TensorFlow Lite Micro
4. Membaca dan mendekuantisasi window hasil rekonstruksi
5. Menghitung Mean Squared Error (MSE)
6. Membandingkan MSE dengan threshold
7. Mengklasifikasikan window sebagai `NORMAL` atau `ANOMALY`

Contoh hasil:

```text
MSE       = ...
Threshold = 0.09787860
Status    = NORMAL
```

Jika error rekonstruksi lebih besar dari threshold, hasilnya:

```text
Status    = ANOMALY
```

Jika error rekonstruksi lebih kecil atau sama dengan threshold, hasilnya:

```text
Status    = NORMAL
```

## Cara Kerja Deteksi Anomali

Model yang di-deploy adalah Autoencoder.

```text
Input window (24 jam x 8 fitur)
      |
      v
Scaling dan kuantisasi int8
      |
      v
Autoencoder
      |
      v
Window hasil rekonstruksi
      |
      v
Hitung MSE
      |
      v
Bandingkan dengan Threshold
      |
      +---- MSE <= Threshold ----> NORMAL
      |
      +---- MSE > Threshold -----> ANOMALY
```

Autoencoder dilatih untuk merekonstruksi pola harian data greenhouse yang normal. Ketika sebuah window berbeda jauh dari pola yang dipelajari, error rekonstruksinya naik.

Skenario anomali sintetis untuk evaluasi: lonjakan suhu, CO2 ekstrem, sensor macet, RH jenuh di siang hari, dan risiko jamur di siang hari.

## Catatan Penting

- File `.ipynb` adalah Jupyter Notebook untuk Google Colab.
- Jalankan notebook berurutan: feature analysis, training, lalu konversi TFLite.
- Unggah seluruh folder `greenhouse-anomaly-edgeai` ke Google Drive supaya path di notebook valid.
- Folder `datasets/clean` dan folder `models` dihasilkan oleh notebook. Salinan siap pakai sudah ada di repositori ini.
- Scaler harus di-fit hanya pada data latih. Jangan fit ulang pada data baru.
- Threshold berlaku pada data ter-scale. Selalu scale input sebelum menghitung MSE.
- Model int8 membutuhkan kuantisasi input dan dekuantisasi output di ESP32.
- Arduino IDE harus terpasang di komputer lokal sebelum deployment ESP32.
- Driver USB atau serial ESP32 yang sesuai harus terpasang jika ESP32 tidak terdeteksi.
- Library `Chirale_TensorFlowLite` dibutuhkan untuk inferensi TensorFlow Lite Micro.
- Jika model dilatih ulang atau `model_type` diganti, jalankan Notebook 3 lagi agar `greenhouse_ae.h` dibuat ulang, dan perbarui threshold di sketch.

## Alur Akhir

```text
GitHub
  |
  | Code > Download ZIP
  v
Ekstrak ZIP
  |
  +--> Google Drive (My Drive)
  |      |
  |      +--> greenhouse-anomaly-edgeai/
  |             |
  |             +--> notebooks/
  |             |      +--> Feature Analysis.ipynb
  |             |      +--> Training Autoencoder.ipynb
  |             |      +--> TFLite Conversion.ipynb
  |             |
  |             +--> datasets/ (dirty, lalu clean)
  |             +--> models/ (.h5, .tflite, scaler, config)
  |
  +--> Komputer Lokal
         |
         +--> Instal Arduino IDE
         |
         +--> Instal Board Package ESP32
         |
         +--> Instal Driver USB atau Serial
         |
         +--> Instal Chirale_TensorFlowLite
         |
         +--> Deployment ESP32
                |
                +--> models/esp32_deploy_tflite/greenhouse_ae.h
                +--> Sketch Arduino (.ino)
                       |
                       v
                Arduino IDE
                       |
                Pilih Board ESP32
                       |
                Pilih Port COM
                       |
                    Compile
                       |
                     Upload
                       |
                Serial Monitor
                       |
                  Input window
                       |
                   Inferensi
                       |
                MSE + Threshold
                       |
              NORMAL / ANOMALY
```

## Informasi Dataset

**Nama dataset:** Dindigul Greenhouse Indoor Dataset (2024 dan 2025)

**Aplikasi:** Deteksi anomali Edge Machine Learning untuk iklim greenhouse

**Model utama:** Autoencoder (Dense)

**Target deployment:** ESP32

**Input:** Window 24 jam dengan 8 fitur (suhu, kelembapan, CO2, radiasi matahari, VPD, titik embun, leaf wetness proxy, day night flag)

**Keluaran inferensi:** Window hasil rekonstruksi, MSE, perbandingan dengan threshold, dan status anomali
