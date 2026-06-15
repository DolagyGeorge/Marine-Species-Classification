# Marine-Species-Classification

## Overview
This repository contains a deep learning pipeline for classifying marine species. The project utilizes the FathomNet API and COCO annotations to dynamically download, preprocess, and classify marine images. A custom Convolutional Neural Network (CNN) is built from scratch using TensorFlow and Keras to learn and predict the species categories.

## Features
* **Automated Data Ingestion:** Downloads images dynamically using the official `fathomnet` Python API based on COCO annotations (`train.json`).
* **Robust Preprocessing:** Handles out-of-bounds bounding boxes via dynamic coordinate clamping and precise cropping, ensuring high-quality input data.
* **Data Augmentation:** Implements random horizontal/vertical flipping, brightness, and contrast adjustments to improve model generalization.
* **Custom CNN Architecture:** A streamlined sequential CNN model equipped with Batch Normalization and Dropout for robust learning and regularization.
* **Optimized Training:** Utilizes `tf.data` API for highly efficient data pipelining and callbacks like `ModelCheckpoint` and `ReduceLROnPlateau` to ensure the best weights are saved.

## Dataset
The dataset is dynamically sourced from [FathomNet](https://fathomnet.org/). The notebook parses a `train.json` COCO annotation file, automatically identifies the top 5 most common marine species classes, and downloads up to 400 images per class (resulting in a dataset of up to 2,000 images). 

*Images are automatically filtered to remove broken links or empty files before building the dataset.*

## Model Architecture
The custom CNN is built from scratch with the following structure:

1.  **Block 1:** Conv2D (32 filters, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D (2x2)
2.  **Block 2:** Conv2D (64 filters, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D (2x2)
3.  **Block 3:** Conv2D (128 filters, 3x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D (2x2)
4.  **Classification Head:** Flatten $\rightarrow$ Dense (256 units) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout (0.5) $\rightarrow$ Dense (Softmax output for 5 classes)

**Training Strategies:**
* **Optimizer:** Adam 
* **Callbacks:** * `ModelCheckpoint`: Saves the best model based on validation accuracy.
    * `ReduceLROnPlateau`: Decreases the learning rate dynamically if validation loss stops improving.

## Requirements & Installation

To run this notebook locally or on a cloud environment, you will need Python 3.x and the following dependencies:

```bash
pip install tensorflow numpy pycocotools fathomnet requests tqdm
