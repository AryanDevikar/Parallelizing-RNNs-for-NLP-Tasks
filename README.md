# Parallel Recurrent Architectures for Scalable NLP Tasks

**Course:** 16:954:577:01 Statistical Software Final Project
**Team 1:** Aryan Nitin Devikar (ad2069), Chaitanya Deogaonkar (cmd517), Sahil Parab (sp2627)

---

## Overview

This project investigates whether simplified, parallelizable recurrent neural networks can match the performance of Transformers while retaining efficiency advantages. Minimal variants of traditional RNNs — **minLSTM** and **minGRU** — were implemented and evaluated based on the architecture proposed by [Feng et al. (2024)](https://doi.org/10.48550/arXiv.2410.01201) in *"Were RNNs All We Needed?"*.

The key innovation involves removing hidden state dependencies from gating mechanisms, enabling parallel training via **parallel scan algorithms** while maintaining linear complexity **O(n)**.

---

## Tasks

### 1. Language Modeling
- **Dataset:** WikiText-103 (~103M tokens, 28,475 Wikipedia articles)
- **Tokenizer:** GPT-2 BPE (vocab size: 50,257)
- **Splits:** ~1.8M train / ~3,760 validation / ~4,358 test samples
- **Task:** Next-token prediction (probability distribution over vocabulary)

### 2. Text Classification
- **Dataset:** AG News Corpus (120,000 articles)
- **Classes:** World, Sports, Business, Technology (balanced, 25% each)
- **Splits:** 96,000 train / 24,000 validation / 7,600 test
- **Task:** 4-class news article categorization

---

## Models

### Baseline — Sequential RNNs (RNN, LSTM, GRU)
- PyTorch built-in implementations with hidden state dependencies
- Trained sequentially via Backpropagation Through Time (BPTT)
- Hyperparameters: embedding dim 200–256, hidden dim 256–512, 2 layers, dropout 0.3–0.5

### Proposed — Minimal Architectures (minLSTM, minGRU)
- Hidden state dependencies removed from gating mechanisms
- minGRU additionally removes the reset gate entirely
- Parallelizable via parallel scan during training
- Hyperparameters optimized via grid search: embedding dim 256–384, hidden dim 512, **3 layers**, dropout 0.2–0.3, higher learning rate (1e-3 to 1.5e-3)

### Transformer Baselines
| Model | Task | Parameters |
|---|---|---|
| GPT-2 Small | Language Modeling | 124.4M |
| DistilBERT | Classification | 66.9M |
| Longformer | Classification | 148.6M |

All transformer models were **fine-tuned** (not trained from scratch) on the respective datasets.

---

## Results

### Language Modeling

| Model | Perplexity | CE Loss | Memory | Training Time | Parameters |
|---|---|---|---|---|---|
| RNN | 103.73 | 4.642 | 2,300 MB | 8h 52m | 39.5M |
| LSTM | 73.21 | 4.293 | 2,661 MB | 12h 33m | 42.3M |
| GRU | 71.66 | 4.271 | 2,670 MB | 12h 37m | 41.4M |
| **minLSTM** | **1.74** | **0.556** | 3,148 MB | **6h** | 41.0M |
| **minGRU** | **1.54** | **0.435** | 5,913 MB | **8h 42m** | 41.0M |
| GPT-2 Small | 1.113 | 0.113 | 9,734 MB | 41h 6m | 124.4M |

### Text Classification

| Model | Accuracy | F1-Score | Memory | Training Time | Parameters |
|---|---|---|---|---|---|
| RNN | 25.00% | 0.1005 | 306 MB | 21m 16s | 11.6M |
| LSTM | 92.74% | 0.9264 | 418 MB | 21m 13s | 18.9M |
| GRU | 92.34% | 0.9233 | 442 MB | 21m | 16.5M |
| **minLSTM** | **91.62%** | **0.9162** | **362 MB** | **17m 37s** | 12.6M |
| **minGRU** | **91.42%** | **0.9142** | **313 MB** | **17m 26s** | 11.7M |
| DistilBERT | 94.45% | 0.9444 | 1,122 MB | 1h 45m | 66.9M |
| Longformer | 94.88% | 0.9488 | 2,436 MB | 6h 30m | 148.6M |

---

## Key Findings

- **~50× lower perplexity** for minLSTM/minGRU vs. traditional RNNs on language modeling, with **50% less training time**
- **Comparable classification accuracy** to traditional RNNs/LSTMs with **25% less training time** and **33% less memory**
- For the AG News task, minimal architectures are a **better alternative to Transformers** — similar performance at a fraction of the compute cost
- **Layer stacking is crucial**: depth compensates for per-layer simplicity (single-layer models underperform dramatically; 3-layer models excel)
- GPT-2 outperforms all recurrent models on language modeling, but benefits from 3× more parameters and prior exposure to the dataset during pre-training

---

## Implementation Details

- **Optimizer:** AdamW with weight decay 0.01
- **LR Schedule:** Linear warmup + cosine decay
- **Regularization:** Dropout (0.2–0.3), gradient clipping at norm 1.0, early stopping
- **Precision:** Mixed precision training (FP16) with GradScaler
- **Hardware:** 16GB GPUs (vs. 80GB A100s in the original paper)

---

## Reference

Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H. (2024). *Were RNNs All We Needed?* arXiv:2410.01201. https://doi.org/10.48550/arXiv.2410.01201
