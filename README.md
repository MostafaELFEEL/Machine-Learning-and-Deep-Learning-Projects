# Machine Learning and Deep Learning Projects
# 1-Gradient Descent Optimization Techniques:

## Overview
This repository delves into the application of various Gradient Descent methods to identify the local minimum of a given function. The project aims to solve a set of non-linear equations by leveraging different optimization techniques, ultimately highlighting the efficacy of Gradient Descent algorithms in minimizing a specified objective function.

## Problem Statement
The primary objective revolves around finding solutions for a set of non-linear equations, as depicted below:

![Equations](https://user-images.githubusercontent.com/106331831/236287269-b4d2a776-55e4-44fb-bbd3-709e64ab1f70.png)

These equations are reformulated to minimize the suggested objective function:

![Objective Function](https://user-images.githubusercontent.com/106331831/236287358-dc769fc9-867d-47be-ad38-269cadb58df9.png)

## Gradient Descent Methods
The project explores three distinct Gradient Descent techniques to optimize the objective function:
1. **Conventional Gradient Descent Method**: A traditional approach to iteratively update parameters.
2. **Newton-Raphson’s Method**: Utilizes second-order derivatives to refine the optimization process.
3. **Line Search Gradient Descent (Steepest)**: Focuses on determining the optimal step size for efficient convergence.

## Results and Visualization
The outcomes of the Gradient Descent methods are showcased through detailed results, illustrating the convergence behavior and efficiency of each technique. For a visual representation of the results, refer to the following:

![Results Visualization](https://github.com/MostafaELFEEL/Machine-Learning/assets/106331831/cfad1c66-77b5-4ad9-8404-960fa40e41e9)

## Implementation Steps
1. Clone the repository to your local machine.
2. Navigate to the project directory and review the provided codebase.
3. Execute the Python scripts corresponding to each Gradient Descent method to observe the optimization outcomes.
4. Analyze the generated results to gain insights into the convergence patterns and effectiveness of each technique.

## Conclusion
This repository serves as a comprehensive guide to understanding and implementing various Gradient Descent optimization techniques. By exploring different methods, the project sheds light on their respective advantages and limitations, paving the way for informed decision-making in optimization tasks.

For a deeper understanding and detailed insights, explore the code snippets, and visualize the results presented in this repository.


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


# 3-CNN-Based Linear Regression for Car Thrust and Steering Prediction with pytorch:

## Overview
This repository presents a Convolutional Neural Network (CNN) approach tailored for linear regression tasks focused on predicting thrust and steering values based on front-view images of a car. Leveraging the power of CNNs, the model is trained using a dataset comprising front-view car images paired with corresponding thrust and steering measurements.

## Problem Statement
The primary goal is to develop a predictive model capable of estimating thrust and steering values solely based on visual inputs. By analyzing front-view images of a car, the CNN model aims to accurately determine the optimal thrust and steering commands, thereby facilitating efficient autonomous navigation.

## Dataset
The dataset utilized for training consists of:
- Front-view images of cars captured under diverse environmental conditions.
- Corresponding thrust and steering measurements associated with each image.

## CNN Model Architecture
The CNN model architecture is designed to extract intricate features from the input images, subsequently predicting thrust and steering values. The model comprises multiple convolutional and pooling layers, followed by fully connected layers to facilitate regression-based predictions.

## Implementation Steps
1. **Dataset Preparation**: Ensure the dataset containing front-view car images and associated thrust-steering data is organized appropriately.
2. **Model Training**: Execute the provided Python script to initiate the CNN model training process. Adjust hyperparameters if necessary to optimize model performance.
3. **Evaluation**: After training, evaluate the model's performance using a separate validation dataset to assess its accuracy and reliability in predicting thrust and steering values.
4. **Deployment**: Once satisfied with the model's performance, deploy it in a real-world scenario, enabling real-time prediction of thrust and steering commands based on front-view car images.

## Results and Visualization
Visualize the model's predictions and performance metrics through comprehensive graphs and charts included in the repository. Analyzing these visualizations will provide insights into the model's efficacy and potential areas for improvement.

![image](https://github.com/MostafaELFEEL/Machine-Learning/assets/106331831/15c34b44-e94f-4aa4-913f-b7a001a665f3)

Video Link: https://youtu.be/7oRD97KmfC4

## Conclusion
This project showcases the application of CNNs in addressing linear regression challenges, specifically predicting thrust and steering values based on front-view car images. By leveraging deep learning techniques, the model demonstrates promising results, emphasizing the significance of visual data in autonomous navigation systems.

For detailed implementation instructions, code snippets, and performance evaluations, navigate through the repository and explore the provided resources.
