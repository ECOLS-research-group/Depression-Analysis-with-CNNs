# Depression-Analysis-with-CNNs

This repository contains Python implementations for analyzing depression from social media text data using traditional, hybrid, and new deep learning approaches. It includes methods for encoding text, transforming data into visual representations, and classifying data using CNNs, BERT, LSTMs, and other algorithms.

Copyright (c) 2024, ECOLS - All rights reserved.

---

## Version 1.0

The student **Anuraag Raj** wrote the code for **Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Detecting Depression on Social Media**, with contributions from **Dr. Anuraganand Sharma**. The code was written entirely in **Python**.

**Paper**: *Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Detecting Depression on Social Media*. Paper submitted to the **IEEE Transactions on Artificial Intelligence**.  
**Authors**: Anuraag Raj and Anuraganand Sharma.

---

## Folder Structure

The repository is structured into the following folders:

### **1. Data**

Contains the dataset files and image transformations used for classification tasks:
- **Files**:
  - `depression_dataset_reddit_cleaned.csv`: Preprocessed dataset of social media posts.
  - `merged_tensors_with_labels.csv`: Encoded vectors using BERT.
- **Subfolders**:
  - `AlgX3_64x64_merged_tensors_with_labels`: Contains AlgX3-transformed images (64x64).
  - `Bargraphs_merged_tensors_with_labels`: Contains bar graph images.
  - `Heatmaps_merged_tensors_with_labels`: Contains heatmap images.

### **2. Traditional Approaches**

Contains implementations of traditional approaches for depression detection:
- **File**: 
  - `Depression_Detection_from_Text_Using_LSTM,_SVM,_RF,_and_1D_CNN.ipynb`: Implements LSTM, SVM, RF, and 1D CNN models for classification tasks.

### **3. Hybrid Approaches**

Contains implementations combining BERT with other models:
- **File**:
  - `BERT_Encoding_Using_LSTM,_SVM,_RF,_and_1D_CNN.ipynb`: Combines BERT embeddings with LSTM, SVM, RF, and 1D CNN models.

### **4. New Approaches**

Contains implementations of advanced deep learning techniques:
- **Subfolder**: `image_transformation`
  - **File**: `Data_Transformation.ipynb`: Transforms encoded data into visual representations (heatmaps, bar graphs, and histograms).
- **Files**:
  - `bert_only.ipynb`: Implements classification using only BERT embeddings.
  - `BERT_v_Autoencoder.ipynb`: Compares BERT-based models with autoencoders.
  - `GSGD_CNN.ipynb`: Uses Guided Stochastic Gradient Descent (GSGD) to optimize 2D CNN models for image-based classification.
  - `model.py`: Defines CNN architecture.
  - `train.py`: Contains training functions.
  - `ViT.ipynb`: Implements Vision Transformers for depression detection tasks.

---

## Key Features

- **Image Transformation Techniques**:
  - Transform encoded vectors into images such as heatmaps, bar graphs, and histograms.
  - Organized into respective folders under the `data` directory.

- **Traditional Approaches**:
  - Implements models like LSTM, SVM, RF, and 1D CNN for textual data.

- **Hybrid Approaches**:
  - Combines BERT embeddings with models such as LSTM and CNNs for enhanced performance.

- **New Approaches**:
  - Advanced models like Vision Transformers (ViT) and GSGD-optimized 2D CNNs.
  - **GSGD**: A novel optimization algorithm tailored for 2D CNN-based classification tasks.

---

## Requirements

Ensure you have the following Python packages installed:
- **Python 3.x**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ECOLS-research-group/Depression-Analysis-with-CNNs.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Depression-Analysis-with-CNNs
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Step 1: Data Transformation

Run the `Data_Transformation.ipynb` file in the `new_approaches/image_transformation` folder to:
- Generate heatmaps, bar graphs, and histograms from encoded data (`merged_tensors_with_labels.csv`).
- Transformed images are saved in organized folders under `data`.

### Step 2: Traditional Approaches

Run the `Depression_Detection_from_Text_Using_LSTM,_SVM,_RF,_and_1D_CNN.ipynb` file to:
- Train traditional models such as LSTM, SVM, RF, and 1D CNN using textual data.

### Step 3: Hybrid Approaches

Run the `BERT_Encoding_Using_LSTM,_SVM,_RF,_and_1D_CNN.ipynb` file to:
- Train hybrid models combining BERT embeddings with traditional classifiers.

### Step 4: New Approaches

Run the respective files in the `new_approaches` folder:
- `GSGD_CNN.ipynb`: Train and optimize a 2D CNN using the GSGD algorithm on transformed image data.
- `bert_only.ipynb`: Train BERT-based models for classification.
- `BERT_v_Autoencoder.ipynb`: Compare BERT and autoencoder-based models.
- `ViT.ipynb`: Train Vision Transformers for depression detection.

---

## Parameters for GSGD

GSGD (Guided Stochastic Gradient Descent) is used for optimizing 2D CNNs:
- **Major**:
  - `lr`: Learning rate.
  - `rho`: Neighborhood size for consistent batches.
  - `batch_size`: Batch size during training.
- **Minor**:
  - `revisit_batch_num`: Consistent batches revisited during weight updates.
  - `verification_set_num`: Validation set for batch consistency checks.

---

## Google Colab Integration

To run the code in Google Colab:
1. Upload required files (`depression_dataset_reddit_cleaned.csv` and `merged_tensors_with_labels.csv`) to the Colab environment.
2. Open any `.ipynb` file in Colab.
3. Install required libraries:
    ```bash
    !pip install tensorflow keras pandas matplotlib seaborn scikit-learn
    ```
4. Execute the cells step-by-step to preprocess data, train models, and evaluate performance.

---

## Acknowledgments

This project builds upon work in NLP and deep learning, particularly in text and image-based classification tasks.

---

## Contributors

- **Anuraag Raj** (Primary Developer)
- **Associate Prof. Anuraganand Sharma** (Research Supervisor)
