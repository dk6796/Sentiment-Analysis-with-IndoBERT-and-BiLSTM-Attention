# Indonesian Sentiment Analysis — BiLSTM+Attention vs IndoBERT

Proyek ini mengimplementasikan dan membandingkan dua pendekatan analisis sentimen untuk teks Bahasa Indonesia:
1. **BiLSTM + Self-Attention** — model berbasis recurrent neural network yang dibangun dari awal
2. **IndoBERT Fine-tuning** — fine-tuning model pre-trained `indobenchmark/indobert-base-p1`

Kedua model dievaluasi pada dataset dengan tiga kelas sentimen: **Positif**, **Netral**, dan **Negatif**.

## Dataset
Dataset berformat TSV (Tab-Separated Values) dengan dua kolom tanpa header:
| Kolom | Deskripsi |
|---|---|
| `text` | Teks ulasan / kalimat dalam Bahasa Indonesia |
| `label` | Sentimen: `positive`, `neutral`, atau `negative` |

Dataset dibagi menjadi tiga split: `train_preprocess.tsv`, `valid_preprocess.tsv`, dan `test_preprocess.tsv` dapat dilihat dan unduh di folder DATASETS.
