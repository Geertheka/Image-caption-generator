# 🖼️ Image Caption Generator — ResNet-50 + LSTM

Automatically generate natural language captions for images using a CNN encoder and LSTM decoder architecture, trained on the Flickr8K dataset.

## Overview

This project implements an **image captioning system** that takes a raw image as input and produces a descriptive English sentence. It combines:

- **ResNet-50** (Convolutional Neural Network) as the image encoder — extracts rich visual features from the input image
- **LSTM** (Long Short-Term Memory) as the text decoder — generates a caption word by word, conditioned on the visual features

The model is trained end-to-end on the **Flickr8K** dataset, which contains 8,000 images each paired with 5 human-written captions.

**Example:**

| Input Image | Generated Caption |
|-------------|-------------------|
| A dog on a field | *"A dog is running through the grass"* |
| Two kids playing | *"Two children are playing in the park"* |

---

## Architecture

```
Input Image (224×224×3)
        │
        ▼
   ResNet-50
   (pretrained on ImageNet, classifier head removed)
        │
        ▼
  Feature Vector
   (avg pool output)
        │
        ▼
   Dense Layer
  (Image Embedding)
        │
        └──────────────────────┐
                               ▼
Raw Caption ──► Preprocessing ──► Tokenizer ──► Embedding Table
                                                      │
                                              Word Embeddings
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │   LSTM Decoder  │
                                             │                 │
                                             │  t=0 : image    │
                                             │  embedding      │
                                             │  seeds h0, c0   │
                                             │                 │
                                             │  t≥1 : word     │
                                             │  embedding +    │
                                             │  h_{t-1}        │
                                             └────────┬────────┘
                                                      │
                                                      ▼
                                             Dense + Softmax
                                          (vocab_size probabilities)
                                                      │
                                                      ▼
                                            Generated Caption
                                     "A dog running in the grass"
```

### Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Image encoder | ResNet-50 (pretrained) | Transfer learning from ImageNet gives strong visual features without training from scratch |
| Projection | Dense layer | Projects 2048-dim ResNet output into the same 256-dim space as word embeddings |
| Text decoder | LSTM | Handles sequential dependencies; remembers earlier words to maintain coherent grammar |
| Embedding size | 256 | Balances representational capacity with training speed on Flickr8K vocabulary (~8K words) |
| Training strategy | Teacher forcing | Feeds ground-truth previous word during training for stable convergence |

---

## Dataset

**Flickr8K** — 8,000 images collected from Flickr, each annotated with 5 different human-written captions (40,000 captions total).

| Split | Images | Captions |
|-------|--------|----------|
| Train | 6,000  | 30,000   |
| Validation | 1,000 | 5,000 |
| Test  | 1,000  | 5,000    |

## Project Structure


## Results

Evaluated using **BLEU score** on the Flickr8K test set (1,000 images):

| Metric | Score |
|--------|-------|
| BLEU-1 | ~0.764|
| BLEU-2 | ~0.538|
| BLEU-3 | ~0.384|
| Overall BLEU | ~0.78|

> Note: Scores will vary based on training duration, hardware, and random seed. The above are approximate values for a 20-epoch run.

### Sample Outputs

```
Image: dog-running.jpg
Ground truth : "a brown dog is running through the grass"
Generated    : "a dog is running through the green grass"

Image: kids-park.jpg
Ground truth : "two young children are playing on a slide"
Generated    : "two children are playing in the park"
```

---

## How It Works

### 1. Image Encoding
A raw image is resized to 224×224 and passed through ResNet-50. The final average pooling layer outputs a feature vector which is then projected by a Dense layer into a 256-dimensional **image embedding**.

### 2. Text Preprocessing
All captions are lowercased, stripped of punctuation, and wrapped with `[start]` and `[end]` tokens. Words are mapped to integer indices via a tokenizer built from the training vocabulary.

### 3. Word Embeddings
Each token index is looked up in a trainable embedding table of shape `(vocab_size × 256)`, producing a 256-dimensional vector per word.

### 4. LSTM Decoding
- At `t=0`: the image embedding is fed into the LSTM, seeding its hidden state `h_0` and cell state `c_0` with visual context.
- At `t≥1`: the embedding of the previous word is fed in alongside the carried hidden state. The LSTM outputs a new hidden state which is passed through a Dense + Softmax layer to produce a probability distribution over the vocabulary. The highest-probability token is selected as the next word.
- This loop continues until the `[end]` token is predicted.

### 5. Teacher Forcing (Training only)
During training, the ground-truth previous word is fed in at each step (rather than the model's own prediction). This stabilises training gradients and speeds up convergence.

---

## Limitations

- Trained only on Flickr8K — captions may be generic for unusual or domain-specific images
- No attention mechanism — the LSTM receives the image context only at `t=0`, so spatial details can be lost in longer captions
- Vocabulary is fixed at training time — out-of-vocabulary words are replaced with `<UNK>`
- BLEU score is an imperfect metric and does not fully capture caption quality



---
