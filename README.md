# Depression-Analysis-with-CNNs

This repository contains Python implementations of 1D and 2D Convolutional Neural Networks (CNNs) designed for analyzing depression from text data. It also includes custom optimization algorithms and various image transformation techniques, such as heatmaps and bar graphs, to visualize and interpret the model results effectively.

Copyright (c) 2024, ECOLS - All rights reserved.

## Version 1.0

The student **Anuraag Raj** wrote the code for **Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Depression Detection on Social Media**, with contributions from **Dr. Anuraganand Sharma**. The code was written entirely in **Python**.

**Paper**: *Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Depression Detection on Social Media*. Paper submitted at the **IEEE Transactions on Artificial Intelligence**.  
**Authors**: Anuraag Raj and Anuraganand Sharma.

## Overview

This repository provides implementations of 1D and 2D CNNs, as well as custom optimization techniques, aimed at analyzing depression from social media text data. Key processes include image transformations from encoded vectors, training and evaluating CNN models, and employing advanced algorithms for classification.

### Key Features

- **Image Transformation Techniques:** 
  - Generate heatmaps and bar graph images from encoded vectors using the `Data_Transformation.ipynb` file.
  - Images are saved in appropriate folders for use in classification tasks.

- **1D and 2D CNN Models:** 
  - Train CNNs using transformed images (heatmaps, histograms, and bar graphs) in the `CNNs.ipynb` file.
  - Includes visualizations and comprehensive performance reports.

- **Custom Optimization with GSGD Algorithm:** 
  - Implemented in the `GSGD_updated.ipynb` file.
  - Optimizes and classifies using transformed images with a novel approach for enhanced accuracy.

- **Support Vector Machine (SVM) vs. Random Forest (RF):**
  - Compare classification performance of encoded vector data using the `SVM_v_RF.ipynb` file.

## Requirements

To run the code in this repository, ensure that you have the following Python packages installed:

- **Python 3.x**
- **TensorFlow**: For building and training the CNN models.
- **Keras**: For high-level neural networks API running on top of TensorFlow.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting and visualizing data.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For machine learning utilities and metrics.

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

## Usage

### Step 1: Data Transformation

Run the `Data_Transformation.ipynb` file to:
- Generate heatmaps and bar graph images from encoded vectors stored in `merged_tensors_with_labels.csv`.
- Images will be saved in organized folders for use in subsequent classification tasks.

### Step 2: Classification Using CNNs

Run the `CNNs.ipynb` file to:
- Use the transformed images (histograms, bar graphs, and heatmaps) for classification.
- Train 1D and 2D CNN models and generate performance visualizations and detailed reports.

### Step 3: GSGD Optimization

Run the `GSGD_updated.ipynb` file to:
- Optimize classification tasks using the GSGD algorithm.
- Evaluate the performance on transformed images, leveraging advanced optimization techniques.

### Step 4: SVM vs. Random Forest

Run the `SVM_v_RF.ipynb` file to:
- Perform classification on encoded vector data using Support Vector Machines (SVM) and Random Forest (RF).
- Compare their performance on the depression dataset.

## Datasets

This project uses the following datasets:
- **`depression_dataset_reddit_cleaned.csv`**: Original dataset retrieved from Kaggle.
- **`merged_tensors_with_labels.csv`**: Encoded data using BERT.
- **Images**: Generated and saved using the `Data_Transformation.ipynb` file. The images are organized into folders based on their transformation type (e.g., heatmaps, bar graphs, histograms) and class (`0/` and `1/`).

## Google Colab Integration

To run the code in Google Colab:

1. Upload the required dataset files (`depression_dataset_reddit_cleaned.csv` and `merged_tensors_with_labels.csv`) to the Colab environment.
2. Open any of the `.ipynb` files (`Data_Transformation.ipynb`, `CNNs.ipynb`, `GSGD_updated.ipynb`, or `SVM_v_RF.ipynb`) in Colab.
3. Ensure all required libraries are installed by running:
    ```python
    !pip install tensorflow keras pandas matplotlib seaborn scikit-learn
    ```
4. Execute the cells step-by-step to preprocess data, train models, and evaluate performance.

## Acknowledgments

This project is built upon the work of researchers and developers in the fields of NLP and deep learning, particularly those who developed CNNs and transformer models.

## Programmer

- Anuraag Raj
- Anuraganand Sharma
