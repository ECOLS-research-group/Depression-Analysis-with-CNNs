# Depression-Analysis-with-CNNs
This repository contains Python implementations of 1D and 2D Convolutional Neural Networks (CNNs) designed for analyzing depression from text data. The project explores various image transformation techniques, including histograms, heatmaps, and bar graphs, to visualize and interpret the results of the CNN models.

Copyright (c) 2024, ECOLS - All rights reserved.

## Version 1.0

The student **Anuraag Raj** wrote the code for **Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Depression Detection on Social Media**, with **Dr. Anuraganand** contributing by writing the code in MATLAB. The code was written in **Python** and **MATLAB**.

**Paper**: *Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Depression Detection on Social Media*. Paper submitted at the **IEEE Transactions on Artificial Intelligence**.  
**Authors**: Anuraag Raj and Anuraganand Sharma.

## Overview

This repository provides implementations of 1D and 2D Convolutional Neural Networks (CNNs) aimed at analyzing depression from text data. It includes various image transformation techniques to visualize and interpret the CNN models' performance effectively.

## Key Features
- **1D and 2D CNN Implementations:** 
  - Robust models to analyze and detect patterns associated with depression in social media text data.
  
- **Image Transformation Techniques:** 
  - **Heatmaps:** Visualize the areas of importance in input data.
  - **Histograms:** Display the distribution of model predictions and data characteristics.
  - **Bar Graphs:** Compare performance metrics across different models.

- **Structured Data Organization:**
  - Datasets and output files are organized into distinct folders for easy access and analysis.

- **Research Context:** 
  - This work is grounded in the paper titled *"Revealing Hidden Pain: A Comparative Analysis of Traditional vs. New Deep Learning Approaches for Depression Detection on Social Media,"* authored by Anuraag Raj and Anuraganand Sharma.

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

1. Prepare your dataset and place it in the `data/` directory. The dataset should include:
    - `depression_dataset_reddit_cleaned.csv`: Original data retrieved from Kaggle.
    - `merged_tensors_with_labels.csv`: Encoded data using BERT.
    - Images located in `data/AlgX3_64x64_merged_tensors_with_labels/` that utilize a histogram approach, organized into `0/` and `1/` folders.
    - The `data/bargraphs/` utilize a bar graph approach, organized into `0/` and `1/` folders.
    - The `data/heatmaps/` utilize a heatmap approach, organized into `0/` and `1/` folders.
2. Run the training scripts located in the `py/` directory:
    ```bash
    python py/bargraph_cnn_1d.py
    python py/bargraph_cnn_2d.py
    python py/histogram_cnn_1d.py
    python py/histogram_cnn_2d.py
    python py/heatmap_cnn_1d.py
    python py/heatmap_cnn_2d.py
    ```

## Google Colab Integration

To run the code in Google Colab:

1. Open the notebook file in the `ipynb/` folder:
    - `Colab_Notebook.ipynb`

2. Follow these steps to set up the notebook in Colab:
    - Upload the dataset files (`depression_dataset_reddit_cleaned.csv` and `merged_tensors_with_labels.csv`) to the Colab environment.
    - Ensure all required libraries are installed in the Colab environment by running:
      ```python
      !pip install torch transformers scikit-learn matplotlib
      ```
    - Run the cells in the notebook to preprocess data, train models, and evaluate performance.

## Datasets

This project uses the following datasets:
- **`depression_dataset_reddit_cleaned.csv`**: Original dataset retrieved from Kaggle.
- **`merged_tensors_with_labels.csv`**: Encoded data using BERT.
- **Images**: Located in the `data/AlgX3_64x64_merged_tensors_with_labels/` directory. The images are organized into `0/` and `1/` folders, representing different classes for the histogram approach.


## Acknowledgments

This project is built upon the work of researchers and developers in the fields of NLP and deep learning, particularly those who developed CNNs and transformer models.

## Programmer

- Anuraag Raj
- Anuraganand Sharma

