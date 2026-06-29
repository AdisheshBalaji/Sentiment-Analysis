# Sentiment Analysis with Attention Mechanisms

Binary sentiment classification on the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) using LSTM-based models augmented with various attention mechanisms.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Attention Mechanisms](#attention-mechanisms)
- [Training](#training)
- [Results](#results)
- [Streamlit Demo App](#streamlit-demo-app)
- [Setup & Usage](#setup--usage)

---

## Overview

This project explores how different **attention mechanisms** (Bahdanau, Luong Dot, Luong Concat, Luong General) interact with different **sequence models** (Vanilla RNN, Vanilla LSTM, Bidirectional RNN, Bidirectional LSTM) for sentiment classification. The best-performing model is then served via a **Streamlit web app** that visualises attention weights over input tokens.

---

## Project Structure

```
Session_3_Task/
├── codes/
│   ├── main.ipynb            # Main training notebook
│   ├── app.py                # Streamlit inference app
│   ├── utils.py              # Helper utilities
│   ├── embedding_matrix.pt   # Pre-built GloVe embedding matrix
│   ├── word2idx.pt           # Word-to-index mapping
│   ├── idx2word.pt           # Index-to-word mapping
│   ├── glove.6B.*.txt        # GloVe pre-trained vectors (50/100/200/300d)
│   └── attention/
│       ├── bahdanau.py       # Bahdanau (additive) attention
│       ├── luongdot.py       # Luong Dot-product attention
│       ├── luongconcat.py    # Luong Concat attention
│       └── luonggeneral.py   # Luong General attention
├── models/
│   ├── Vanilla LSTM with Bahdanau.pth
│   ├── Vanilla LSTM with Luong General.pth
│   ├── BiLSTM with Bahdanau.pth
│   └── BiLSTM with LuongConcat.pth
├── plots/
│   ├── vanilla_models_accuracy_comparison.png
│   ├── baseline_models_attention_comparison.png
│   ├── model_attention_score_table.png
│   ├── attention_weights.png
│   └── finetuned_models/     # Per-model training curves
├── logs/                     # Per-epoch training logs
└── reports/
    └── report.pdf
```

---

## Dataset

- **Source:** [Stanford IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) via HuggingFace
- **Size:** 50,000 labelled reviews (25k train + 25k test), binary labels (positive / negative)
- **Split:** Combined and re-split into **70 % train / 10 % val / 20 % test** with stratification
- **Preprocessing:**
  - Lowercasing, HTML tag removal, punctuation stripping, whitespace normalisation
  - Vocabulary built from train + test texts (min frequency ≥ 2); special tokens `<PAD>` and `<UNK>`
- **Embeddings:** **GloVe 100-d** (`glove.6B.100d.txt`) — weights initialised from pre-trained vectors and kept trainable during fine-tuning

---

## Models

All models share the same embedding layer (GloVe 100-d, trainable) and a 0.3 dropout layer applied to sequence outputs.

| Model | Architecture |
|---|---|
| **Vanilla RNN** | Single-direction RNN, hidden size 100 |
| **Vanilla LSTM** | Single-direction LSTM, hidden size 100 |
| **Bidirectional RNN** | BiRNN, hidden size 100 per direction |
| **Bidirectional LSTM** | BiLSTM, hidden size 100 per direction |

**With attention**, the final classifier receives `[context_vector ; hidden_state]` instead of just `hidden_state`, doubling (or quadrupling for Bi-models) the input dimensionality to the fully-connected head.

---

## Attention Mechanisms

Four attention variants were implemented and compared:

### Bahdanau (Additive) Attention
Projects encoder outputs **W_enc** and decoder hidden state **W_dec** to a shared attention dimension, then scores them via a learned vector **v**:

```
score(h_i, s_t) = v · tanh(W_enc · h_i + W_dec · s_t)
α = softmax(scores)
context = Σ αᵢ · hᵢ
```

### Luong Dot Attention
Direct dot product between the decoder hidden state and each encoder output (requires equal dimensionality):

```
score(h_i, s_t) = h_i · s_t
```

### Luong General Attention
Introduces a learnable weight matrix **W** between encoder and decoder:

```
score(h_i, s_t) = h_i · W · s_t
```

### Luong Concat Attention
Concatenates encoder and decoder states, then applies a non-linear projection:

```
score(h_i, s_t) = v · tanh(W · [s_t ; h_i])
```

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-2 |
| Loss | Cross-Entropy |
| Batch size | 64 |
| Max epochs | 10 |
| Early stopping | patience = 3 (on val loss) |
| Hidden size | 64 (fine-tuned models) |
| Attention dim | 32 |

The best checkpoint (highest validation accuracy) is saved to `models/`.

---

## Results

### Baseline — No Attention

All four models trained **without** attention, random-weight initialisation only (no GloVe):

![Vanilla Models Accuracy Comparison](plots/vanilla_models_accuracy_comparison.png)

> All vanilla (no-attention, no-GloVe) models hover near **50 % accuracy**, confirming that the gains below come from the combination of pre-trained embeddings and attention.

---

### Attention Comparison — All Model × Attention Combinations

Training and validation accuracy curves across all 16 model + attention combinations (4 architectures × 4 attention types):

![Baseline Models Attention Comparison](plots/baseline_models_attention_comparison.png)

---

### Validation Accuracy Heatmap

Best validation accuracy achieved by each model × attention combination:

![Model Attention Score Table](plots/model_attention_score_table.png)

Key observations:
- **LSTM + Bahdanau** achieves the highest single validation accuracy (**91.40 %**)
- **Luong Dot** performs poorly on RNN and BiRNN architectures (~82 % and ~87 % respectively), while performing comparably on LSTM architectures
- LSTM-family models are consistently more stable than RNN-family models

---

### Fine-tuned Model Test Metrics

The **Vanilla LSTM with Luong General** attention was selected as the final model and evaluated on the held-out test set:

| Metric | Value |
|---|---|
| Test Loss | 0.3315 |
| **Test Accuracy** | **89.85 %** |
| Test Precision | 0.8988 |
| Test Recall | 0.8985 |
| Test F1 Score | 0.8985 |

---

### Training Logs — Fine-tuned Models

| Model | Best Val Acc | Best Epoch |
|---|---|---|
| Vanilla LSTM + Bahdanau | 90.70 % | Epoch 4 |
| Vanilla LSTM + Luong General | **91.12 %** | Epoch 4 |
| BiLSTM + Bahdanau | 90.98 % | Epoch 3 |
| BiLSTM + Luong Concat | 90.62 % | Epoch 5 |

---

### Attention Weights Visualisation

Sample attention heatmap for a test review — the model highlights the words it found most informative when making its prediction:

![Attention over Tokens](plots/attention_weights.png)

The model correctly attends to semantically rich tokens (e.g. *screenplay*, *affleck*, *ben*) while suppressing function words.

---

## Streamlit Demo App

`codes/app.py` provides an interactive UI where you can type any movie review and see:
1. **Predicted sentiment** (Positive / Negative)
2. **Attention bar chart** over the input tokens

**Run locally:**

```bash
cd codes
streamlit run app.py
```

The app loads the **Vanilla LSTM + Luong General** checkpoint by default.

---

## Setup & Usage

### Requirements

```
torch
transformers / datasets  (HuggingFace)
pandas
scikit-learn
matplotlib
streamlit
```

Install with:

```bash
pip install torch datasets pandas scikit-learn matplotlib streamlit
```

### Running the Notebook

Open `codes/main.ipynb` in Jupyter and run all cells sequentially. GloVe vectors (`glove.6B.100d.txt`) must be present in the `codes/` directory.

### Running the Demo App

```bash
cd codes
streamlit run app.py
```

---

## References

- [Bahdanau et al., 2015 — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Luong et al., 2015 — Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Stanford IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
