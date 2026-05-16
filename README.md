# HindiMiniGPT

A decoder-only Transformer (GPT-style) implemented from scratch in PyTorch for Hindi language modeling and sentiment classification.

This project implements a simplified GPT-2 style architecture featuring masked multi-head self-attention, autoregressive text generation, and downstream text classification using a shared Transformer backbone.


###  Data Preprocessing

- Hindi corpus preprocessing
- Training and validation split generation
- Subword tokenizer training using BPE / SentencePiece
- Token encoding and decoding pipeline
- Classification dataset preprocessing
- Label mapping for sentiment classes
- Dataset pair generation for:
  - Language Modeling `(x, y shifted by one token)`
  - Text Classification `(review, sentiment label)`

---

### Decoder Only Transformer & Next Word Prediction

Implemented a simplified GPT-2 style autoregressive language model including:

- Multi-head causal self-attention
- Query, Key, and Value projections
- Causal masking using lower triangular attention masks
- Feedforward MLP layers with ReLU activation
- Layer normalization
- Residual skip connections
- Token and positional embeddings
- Sequential Transformer blocks
- Language modeling head for next-token prediction

The model was trained on the Hindi corpus using Cross Entropy Loss and evaluated using validation perplexity.

A text generation function was implemented to autoregressively generate Hindi text samples.

---

### Text Classification

A classification head was attached to the pretrained GPT backbone for sentiment analysis.

Implemented:
- GPTClassifier architecture
- Final-token feature extraction for classification
- Linear classification head mapping to 3 sentiment classes
- Fine-tuning on text classification dataset
- Validation accuracy evaluation

---

## Architecture Overview

The project implements a simplified GPT-2 style decoder-only Transformer architecture consisting of:

- Token Embedding Layer
- Positional Embedding Layer
- Multi-Head Causal Self-Attention
- Scaled Dot-Product Attention
- Causal Masking using Lower Triangular Attention Matrix
- Feedforward MLP Network with ReLU Activation
- Residual Skip Connections
- Layer Normalization
- Language Modeling Head
- Classification Head

The same MiniGPT backbone is reused for both:
1. Autoregressive Hindi Language Modeling
2. Sentiment Classification

---

## Hyperparameters & Training Configuration

### Tokenizer

| Parameter | Value |
|---|---|
| Tokenizer Type | BPE / SentencePiece |
| Vocabulary Size | 5000 |

### MiniGPT Architecture

| Parameter | Value |
|---|---|
| Context Length (`block_size`) | 512 |
| Embedding Dimension (`n_embd`) | 128 |
| Number of Attention Heads (`n_head`) | 4 |
| Number of Transformer Layers (`n_layer`) | 4 |
| Dropout | 0.1 |
| Vocabulary Size | 5000 |

### Language Model Training

| Parameter | Value |
|---|---|
| Loss Function | Cross Entropy Loss |
| Training Objective | Autoregressive Next Token Prediction |
| Generation Length | 50 Tokens |
| Device | CUDA / CPU |

### Classification Training

| Parameter | Value |
|---|---|
| Number of Classes | 3 |
| Batch Size | 16 |
| Epochs | 40 |
| Learning Rate (Transformer Layers) | 5e-5 |
| Learning Rate (Classification Head) | 5e-4 |
| Early Stopping Patience | 8 |

### Classification Labels

| Label | Class |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

---

## Evaluation Metrics

### Language Modeling
- Validation Loss
- Perplexity

### Text Classification
- Validation Accuracy
- Correctly Predicted Review Samples

---

## Results

| Task | Metric |
|---|---|
| Language Modeling | Validation Perplexity: 3.8038 |
| Text Classification | Validation Accuracy: 89.01% |

---

## Project Structure

```text
HindiMiniGPT/
│
├── hindi_gpt_training.ipynb
├── hindi_gpt_language_model.pth
├── hindi_gpt_classifier.pth
├── generated_hindi_text.txt
├── classification_predictions.txt
├── README.md
```

---

## Installation

```bash
pip install torch sentencepiece numpy pandas matplotlib
```

---

## Usage

Run the notebook:

```bash
jupyter notebook hindi_gpt_training.ipynb
```

---

## Sample Outputs

Generated Hindi text samples are available in:

```text
generated_hindi_text.txt
```

Classification predictions are available in:

```text
classification_predictions.txt
```

---

## Libraries Used

- Python
- PyTorch
- SentencePiece / BPE
- NumPy
- Pandas
- Matplotlib

---

## Deliverables

- Complete implementation code
- Trained language model weights
- Trained classifier weights
- Generated Hindi text samples
- Classification prediction outputs

---

## Team Members

- Ujjwal 
- Rishi
- Praveer
