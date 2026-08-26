<div align="center">

# 🎨 ArtStyle Predictor

**AI-powered art movement classifier and artist recommendation system, built with PyTorch and Streamlit.**

Upload a painting → get its predicted art movement (one of 12 classes) with confidence scores and a short description.
Pick a favorite artist → get similar artists, recommended by a content-based similarity model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C)](https://pytorch.org/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Acknowledgments](#acknowledgments)

---

## Overview

**ArtStyle Predictor** is a two-page Streamlit application:

1. **Style Classifier** (`ArtStyle.py`) — upload an image of a painting and a fine-tuned EfficientNet-B0 model predicts which of 12 art movements it belongs to, along with a confidence score, a short description of the predicted style, and the top-5 most likely classes.
2. **Artist Recommender** (`pages/Artists.py`) — pick an artist from a dropdown and get a list of similar artists, computed from a precomputed content-based similarity matrix built on artist metadata (genre, nationality, era, bio).

The project began as an academic case study and has since gone through a full diagnose-and-fix cycle, documented in detail in [Changelog / Fixes](#changelog--fixes) below — including retraining the classifier from a completely non-functional state.

---

## Features

- 🖼️ **Drag-and-drop image classification** into 12 art movements: Baroque, Cubism, Expressionism, Impressionism, Pop Art, Post-Impressionism, Primitivism, Renaissance, Romanticism, Suprematism, Surrealism, Symbolism.
- 📊 **Confidence scores and top-5 predictions** for every uploaded image.
- 📖 **Style descriptions** pulled from a local CSV, with a link to the relevant Wikipedia article.
- 🎨 **Artist similarity recommendations** based on genre, nationality, era, and biography text.
- 🌐 **Live artist bios and portraits** fetched from Wikipedia at runtime, with a resilient two-stage lookup (see [Changelog](#changelog--fixes)).
- 🧪 **Diagnostic tooling** (`diagnose_artists.py`) to test the Wikipedia lookup for all artists in the dataset and pinpoint failures.

---

## Demo

> Add a screenshot or GIF of the app here once you have one, e.g.:
>
> ```markdown
> ![App screenshot](assets/demo_video.mp4)
> ```

---

## Project Structure

```
art_style_classifier/
├── ArtStyle.py                    # Main Streamlit page — style classifier
├── requirements.txt                # Python dependencies
├── diagnose_artists.py             # Standalone diagnostic for the Wikipedia lookup
├── artifacts/
│   ├── model.pth                   # Trained EfficientNet-B0 classifier weights
│   └── similarity.pkl              # Precomputed artist-similarity matrix
├── assets/
│   ├── landing.png                 # Landing page banner
│   └── gogh.png                    # Artists page banner
├── data/
│   ├── art_style.csv               # Style name, description, Wikipedia link
│   └── artists.csv                 # Artist metadata (genre, nationality, bio, Wikipedia URL, etc.)
├── notebook/
│   ├── modelling.ipynb             # Trains the style classifier
│   └── artists_recommender.ipynb   # Builds the artist similarity matrix
└── pages/
    └── Artists.py                  # Streamlit page — artist recommender
```

Streamlit auto-detects `pages/Artists.py` and adds it as a second page in the sidebar automatically — no manual routing needed.

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone <your-repo-url>
cd art_style_classifier
pip install -r requirements.txt
```

### Run

```bash
streamlit run ArtStyle.py
```

This opens the app at `http://localhost:8501`. Use the sidebar to upload a painting, or switch to the **Artists** page to explore recommendations.

> ⚠️ Run this command from *inside* the `art_style_classifier/` folder. The app uses relative paths (`data/...`, `artifacts/...`, `assets/...`) that assume it's the current working directory.

---

## How It Works

### Style Classifier

```
Uploaded image
   → resize to 128×128, normalize (ImageNet mean/std)
   → EfficientNet-B0 backbone (ImageNet-pretrained, fine-tuned)
   → Linear(1280 → 12) classification head
   → softmax → predicted class + confidence
   → lookup description/Wikipedia link in data/art_style.csv
```

The model class is `SimpleArtClassifier`: an EfficientNet-B0 backbone with its original ImageNet classification head removed, replaced with a single linear layer mapping the 1,280-dimensional pooled feature vector to 12 class logits.

### Artist Recommender

```
Selected artist
   → look up row in data/artists.csv
   → look up precomputed cosine-similarity row in artifacts/similarity.pkl
   → return top-N most similar artists
   → fetch live bio/portrait for each from Wikipedia
```

Similarity is precomputed offline in `notebook/artists_recommender.ipynb` from a combined text profile of each artist's genre, nationality, years active, and biography, vectorized and compared via cosine similarity — the same general approach as a content-based recommender.

The Wikipedia lookup is a two-stage process (see [Changelog](#changelog--fixes) for why): it first tries the artist's *exact* Wikipedia URL already stored in `artists.csv`, and only falls back to a live fuzzy search if that direct lookup fails.

---

## Model Performance

Current `artifacts/model.pth`, evaluated on a held-out test set of 669 images never seen during training:

| Metric | Value |
|---|---|
| **Overall test accuracy** | **66.97%** |
| Random-guess baseline (12 classes) | ~8.3% |
| Best validation accuracy | 68.1% (epoch 10 of 12) |

### Per-class test accuracy

| Class | Test images | Accuracy |
|---|---|---|
| Baroque | 63 | 82.5% |
| Cubism | 18 | 27.8% |
| Expressionism | 54 | 74.1% |
| Impressionism | 141 | 67.4% |
| Pop Art | 18 | 77.8% |
| Post-Impressionism | 32 | 40.6% |
| Primitivism | 47 | 68.1% |
| Renaissance | 145 | 77.9% |
| Romanticism | 34 | 61.8% |
| Suprematism | 18 | 44.4% |
| Surrealism | 34 | 70.6% |
| Symbolism | 65 | 47.7% |

**Strongest classes:** Baroque, Renaissance, Pop Art, Expressionism (all >70%) — these also had the most training images.
**Weakest classes:** Cubism, Suprematism (both had only 84 training images — the smallest classes in the dataset).

---

## Dataset

The classifier is trained on a subset of the [**Best Artworks of All Time**](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) dataset (Kaggle, ikarus777), which organizes images **by artist**, not by style. To get style-labeled training data:

1. Each artist is mapped to one of the 12 target style classes using the `genre` column in `data/artists.csv` (45 of the dataset's 50 artists map cleanly; the remaining 5 have genres outside the 12 target classes and are excluded).
2. Images are capped at 120 per artist and resized to 160×160 to control dataset size.
3. The result — 4,403 images across 12 classes — is split 70% / 15% / 15% into train / validation / test.

| Split | Images |
|---|---|
| Train | 3,078 |
| Validation | 656 |
| Test | 669 |
| **Total** | **4,403** |

---

## Tech Stack

- **App framework:** [Streamlit](https://streamlit.io/)
- **Deep learning:** [PyTorch](https://pytorch.org/), [timm](https://github.com/rwightman/pytorch-image-models) (EfficientNet-B0)
- **Data:** pandas, NumPy
- **Recommender:** scikit-learn (cosine similarity), NLTK
- **Image handling:** Pillow, torchvision

---

## Acknowledgments

- Dataset: [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) (Kaggle, ikarus777)
- Pretrained backbone: [EfficientNet-B0 weights, timm/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) (rwightman)
- Artist bios and images: [Wikipedia](https://www.wikipedia.org/), via the public REST API
