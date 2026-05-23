# 🇮🇩 Indonesian Sentiment Analysis — BiLSTM+Attention vs IndoBERT

Proyek ini mengimplementasikan dan membandingkan dua pendekatan **analisis sentimen** untuk teks Bahasa Indonesia:

1. **BiLSTM + Self-Attention** — model berbasis recurrent neural network yang dibangun dari awal
2. **IndoBERT Fine-tuning** — fine-tuning model pre-trained `indobenchmark/indobert-base-p1`

Kedua model dievaluasi pada dataset dengan tiga kelas sentimen: **Positif**, **Netral**, dan **Negatif**.

---

## 📁 Struktur Repository

```
├── BiLSTM_Attention.ipynb   # Notebook model BiLSTM + Self-Attention
├── IndoBERT.ipynb           # Notebook fine-tuning IndoBERT
├── README.md
└── data/                    # (tidak disertakan) Letakkan dataset di sini
    ├── train_preprocess.tsv
    ├── valid_preprocess.tsv
    └── test_preprocess.tsv
```

---

## 📊 Dataset

Dataset berformat **TSV (Tab-Separated Values)** dengan dua kolom tanpa header:

| Kolom | Deskripsi |
|---|---|
| `text` | Teks ulasan / kalimat dalam Bahasa Indonesia |
| `label` | Sentimen: `positive`, `neutral`, atau `negative` |

Dataset dibagi menjadi tiga split: `train_preprocess.tsv`, `valid_preprocess.tsv`, dan `test_preprocess.tsv`.

---

## 🧠 Model

### 1. BiLSTM + Self-Attention (`BiLSTM_Attention.ipynb`)

Model custom yang dibangun dengan PyTorch dari awal.

**Arsitektur:**
- **Embedding Layer** — representasi kata dengan dimensi 128
- **Bidirectional LSTM** — 2 layer, hidden size 128, menangkap konteks kiri-kanan
- **Self-Attention** — menghitung bobot perhatian untuk setiap token
- **Fully Connected** — dua layer (256 → 64 → 3) dengan aktivasi ReLU
- **Dropout** — regularisasi dengan rate 0.3

**Hyperparameter:**

| Parameter | Nilai |
|---|---|
| Vocab Size | 20.000 |
| Embedding Dim | 128 |
| Hidden Dim | 128 |
| Num Layers | 2 |
| Dropout | 0.3 |
| Max Sequence Length | 128 |
| Batch Size | 32 |
| Max Epochs | 30 |
| Learning Rate | 1e-3 |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Early Stopping | patience=5 |

---

### 2. IndoBERT Fine-tuning (`IndoBERT.ipynb`)

Fine-tuning model pre-trained `indobenchmark/indobert-base-p1` dari Hugging Face.

**Pendekatan:**
- Tokenizer dari `AutoTokenizer` dengan padding & truncation
- `AutoModelForSequenceClassification` dengan 3 label output
- Linear warmup scheduler (10% dari total steps)
- Gradient clipping (`max_norm=1.0`)

**Hyperparameter:**

| Parameter | Nilai |
|---|---|
| Base Model | `indobenchmark/indobert-base-p1` |
| Max Sequence Length | 128 |
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup |

---

## ⚙️ Instalasi

### Prasyarat

- Python 3.8+
- CUDA-compatible GPU (sangat disarankan)

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas scikit-learn matplotlib
```

---

## 🚀 Cara Penggunaan

### 1. Siapkan Dataset

Letakkan file TSV dataset ke dalam folder `data/`:

```
data/
├── train_preprocess.tsv
├── valid_preprocess.tsv
└── test_preprocess.tsv
```

### 2. Konfigurasi Path

Di awal setiap notebook, isi variabel berikut:

```python
DATASETS = "/path/ke/folder/data"   # folder berisi file TSV
SAVE_PATH = "/path/ke/folder/model" # folder untuk menyimpan model terbaik
```

### 3. Jalankan Notebook

Buka dan jalankan sel secara berurutan di Jupyter Notebook atau Google Colab:

```bash
jupyter notebook BiLSTM_Attention.ipynb
# atau
jupyter notebook IndoBERT.ipynb
```

---

## 📈 Output Training

Kedua notebook akan menghasilkan:

- **Log per epoch** — Train/Val Loss dan Accuracy di setiap epoch
- **Model terbaik** tersimpan otomatis berdasarkan validation accuracy:
  - BiLSTM: `best_bilstm_attention.pt`
  - IndoBERT: `best_bert_model.pt`
- **Classification report** pada test set (precision, recall, F1-score per kelas)
- **Grafik** Training vs Validation Accuracy dan Loss

---

## 🔧 Preprocessing Teks

Kedua model menggunakan pipeline preprocessing yang sama:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # hapus karakter khusus
    text = re.sub(r'\s+', ' ', text).strip()      # normalisasi spasi
    return text
```

---

## 📋 Perbandingan Model

| Aspek | BiLSTM + Attention | IndoBERT |
|---|---|---|
| Jenis | Custom dari scratch | Pre-trained + Fine-tuning |
| Parameter | ~Jutaan (ringan) | ~111 Juta |
| Epochs | Hingga 30 (early stop) | 3 |
| Batch Size | 32 | 16 |
| Keunggulan | Cepat, ringan, interpretable | Akurasi tinggi, konteks kaya |
| GPU Memory | Rendah | Tinggi |

---

## 🤝 Kontribusi

Pull request dan issue sangat disambut! Pastikan perubahan kode disertai penjelasan yang jelas.

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).

---

## 🙏 Acknowledgements

- [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) oleh IndoBenchmark
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
