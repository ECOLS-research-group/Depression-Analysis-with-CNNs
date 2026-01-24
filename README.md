# 🧠 Depression Analysis with CNNs

**Revealing Hidden Pain: A Comparative Study of Traditional, Hybrid, and Deep Learning Models for Depression Detection on Social Media**

---

## 📌 Project Overview

This repository contains the complete implementation of the research work:

> **Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Detecting Depression on Social Media**

The study focuses on detecting depression from social media text using:

* **Traditional machine learning models** (LSTM, SVM, RF, 1D CNN)
* **Hybrid models** (BERT combined with classical and neural classifiers)
* **Image-based deep learning models** (2D CNN, Vision Transformer)

A novel contribution of this work is the **conversion of BERT embedding vectors into image representations** (heatmaps, bar graphs, and histogram-like matrices), enabling the application of **2D CNNs and Vision Transformers** for classification.

---

## 👨‍💻 Authors & Roles

* **Anuraag Raj**
  *Primary Programmer & Research Developer*
  Implemented all algorithms, preprocessing pipelines, transformation techniques, deep learning architectures, and experimental evaluations in Python.

* **Dr. Anuraganand Sharma**
  *Research Supervisor & Project Manager*
  Supervised the research methodology, guided experimental design, validated results, and managed the overall research project.

---

## 📄 Publication Information

**Paper Title:**
*Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Detecting Depression on Social Media*

**Journal:**
**IEEE Access (Accepted)**

**Authors:**
Anuraag Raj, Anuraganand Sharma

---

## 📁 Repository Structure

```
DEPRESSION-ANALYSIS-WITH-CNNS/
│
├── data/
│   ├── depression_dataset_reddit_cleaned.csv
│   ├── merged_tensors_with_labels.csv
│   │
│   ├── AlgX3_64x64_merged_tensors_with_labels/
│   │   ├── 0/   (non-depressed class images)
│   │   └── 1/   (depressed class images)
│   │
│   ├── Bargraphs_merged_tensors_with_labels/
│   │   ├── 0/
│   │   └── 1/
│   │
│   └── Heatmaps_merged_tensors_with_labels/
│       ├── 0/
│       └── 1/
│
├── deep_learning_models/
│   ├── bert_only.ipynb
│   ├── BERT_v_Autoencoder.ipynb
│   ├── embedding_to_image_mapping.ipynb
│   ├── GSGD_CNN.ipynb
│   ├── ViT.ipynb
│   ├── model.py
│   └── train.py
│
├── traditional_vs_hybrid_models/
│   ├── traditional_text_models.ipynb
│   └── bert_hybrid_models.ipynb
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Description

### Files

* **`depression_dataset_reddit_cleaned.csv`**
  Preprocessed Reddit posts labeled as depressed (1) or non-depressed (0).

* **`merged_tensors_with_labels.csv`**
  BERT-encoded sentence embeddings along with class labels.

---

## 🖼 Image Representation (Key Innovation)

BERT embeddings are transformed into 2D image representations for CNN and Vision Transformer processing.

### Transformation Types

* **Heatmaps** – visualize embedding intensity distributions.
* **Bar graphs** – represent feature magnitudes.
* **AlgX3 64×64 matrices** – structured reshaping of vectors into spatial grids.

### Class-wise Organization

Images are stored in class-specific folders:

```
Heatmaps_merged_tensors_with_labels/
├── 0/  (non-depressed)
└── 1/  (depressed)
```

This structure enables direct loading using standard image-based deep learning pipelines.

---

## 🧠 Model Categories

### 1. Traditional Models

Located in: `traditional_vs_hybrid_models/traditional_text_models.ipynb`

* LSTM
* Support Vector Machine (SVM)
* Random Forest (RF)
* 1D CNN

These models operate directly on textual or vectorized features.

---

### 2. Hybrid Models

Located in: `traditional_vs_hybrid_models/bert_hybrid_models.ipynb`

* BERT + LSTM
* BERT + SVM
* BERT + RF
* BERT + 1D CNN

These models combine semantic embeddings with classical classifiers.

---

### 3. New Deep Learning Models

Located in: `deep_learning_models/`

* **bert_only.ipynb** – Pure BERT-based classification
* **BERT_v_Autoencoder.ipynb** – Comparison of BERT and autoencoder representations
* **embedding_to_image_mapping.ipynb** – Converts embedding vectors into images
* **GSGD_CNN.ipynb** – 2D CNN optimized using Guided Stochastic Gradient Descent (GSGD)
* **ViT.ipynb** – Vision Transformer-based classification

---

## ⚙️ GSGD Optimization Parameters

Guided Stochastic Gradient Descent (GSGD) is used for optimizing 2D CNN training:

**Major Parameters:**

* `lr` – Learning rate
* `rho` – Neighborhood size for batch consistency
* `batch_size` – Training batch size

**Minor Parameters:**

* `revisit_batch_num` – Number of revisited consistent batches
* `verification_set_num` – Validation set for batch consistency

---

## 📦 Requirements

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

Install using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Workflow

### Step 1: Vector-to-Image Transformation

Run:

```
deep_learning_models/embedding_to_image_mapping.ipynb
```

This converts BERT vectors into heatmaps, bar graphs, and AlgX3 images and saves them under `data/`.

---

### Step 2: Traditional Models

Run:

```
traditional_vs_hybrid_models/traditional_text_models.ipynb
```

---

### Step 3: Hybrid Models

Run:

```
traditional_vs_hybrid_models/bert_hybrid_models.ipynb
```

---

### Step 4: Deep Learning Models

Run any of the following:

* `GSGD_CNN.ipynb`
* `ViT.ipynb`
* `bert_only.ipynb`
* `BERT_v_Autoencoder.ipynb`

---

## ☁️ Google Colab Support

1. Upload required CSV files to Colab.
2. Open the desired `.ipynb` file.
3. Install dependencies:

```bash
!pip install -r requirements.txt
```

4. Execute cells sequentially.

---

## 📜 License & Copyright

Copyright © 2026
**ECOLS Research Group – All Rights Reserved**

---

## 🙏 Acknowledgments

This work builds upon advances in:

* Natural Language Processing (NLP)
* Deep Learning
* Vision-based representation learning

---

## 👥 Contributors

* **Anuraag Raj** – Programmer & Research Developer
* **Dr. Anuraganand Sharma** – Supervisor & Project Manager

---

✨ *This repository accompanies an accepted IEEE Access publication and serves as a reproducible research framework for depression detection from social media.*
