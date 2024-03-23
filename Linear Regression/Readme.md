# CNN-Based Linear Regression for Car Thrust and Steering Prediction with pytorch:

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
