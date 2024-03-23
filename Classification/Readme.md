# 2-Classification Problem Using CIFAR-100 Dataset with TensorFlow:

## Overview
This repository demonstrates a machine learning classification task using the CIFAR-100 dataset. The goal is to classify images into one of the 100 classes using a Multilayer Neural Network (MNN) model implemented with TensorFlow.

## Dataset
The CIFAR-100 dataset is utilized, consisting of 60,000 32x32 color images across 100 classes, with 600 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

## Features
- **Data Visualization**: Visualized sample images from all 100 classes before and after normalization.
- **Data Normalization**: Preprocessed images by subtracting the mean and dividing by the standard deviation.
- **Model Architecture**: Implemented a Multilayer Neural Network with various dense layers.
- **Model Training**: Trained the MNN model using 40,000 training images and validated using 10,000 validation images.
- **Evaluation**: Evaluated model performance on the test set, achieving accuracy metrics.

## Code Structure
- **Imports**: Necessary libraries are imported, including TensorFlow, NumPy, and Matplotlib.
- **Data Loading**: Loaded CIFAR-100 dataset and displayed its shape.
- **Data Visualization**: Sample images were visualized for each class.
- **Normalization**: Images were normalized for better training efficiency.
- **Data Splitting**: Dataset split into training, validation, and test sets.
- **Model Definition**: MNN model architecture is defined using TensorFlow's Sequential API.
- **Model Compilation**: The model is compiled with appropriate loss and optimization techniques.
- **Model Training**: Trained the MNN model and visualized training and validation metrics.

## Results
- **Model Accuracy**: Achieved an accuracy score on the test dataset using the trained MNN model.
- **Performance Plots**: Plotted accuracy and loss graphs for both training and validation data.

![image](https://github.com/MostafaELFEEL/Machine-Learning/assets/106331831/6651e8fa-b30f-4279-8113-8b3afa671d08)
![image](https://github.com/MostafaELFEEL/Machine-Learning/assets/106331831/488cb932-efb9-4f90-96ed-ea5532e1dd1d)

## Usage
1. Clone the repository and navigate to the project directory.
2. Run the provided Python script to execute the entire classification pipeline.
3. Explore the generated graphs to analyze model performance visually.

## Conclusion
This project provides a comprehensive guide to building and evaluating a classification model using TensorFlow and the CIFAR-100 dataset. The results showcase the effectiveness of the implemented MNN architecture for image classification tasks.

For detailed insights, refer to the code snippets and results displayed in this repository.
