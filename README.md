# Pneumonia Image Classification Project

## Overview

This project is focused on the classification of pneumonia from chest X-ray images, leveraging a dataset available on Kaggle. It showcases two different model implementations for this task: a RandomForest classifier from scikit-learn and a Convolutional Neural Network (CNN) approach, providing a comparative insight into machine learning and deep learning techniques for medical image analysis.

## Project Structure

- `model_training.py` - Contains the training workflow for the RandomForest model, including data preprocessing and model serialization.
- `forest.py` - Stores preprocessed test data (`X_test`, `y_test`) used for evaluating the RandomForest model.
- `test.py` - Script for evaluating the RandomForest model's performance on the test data provided in `forest.py`.
- `trial.py` - Demonstrates making predictions on new images with the trained RandomForest model, illustrating its application.

## Dataset

The dataset used in this project is the pneumonia dataset from Kaggle, which includes chest X-ray images labeled as either 'Pneumonia' or 'Normal'. This dataset is instrumental in training the models to differentiate between healthy and pneumonia-affected lung X-rays.

## Setup

Ensure Python 3.6+ is installed on your system. Install the required dependencies to get started with the project:

```bash
pip install numpy scikit-learn joblib torchvision
```

## Overview of the CNN Training Script

This script, `train_cnn.py`, implements a binary classification Convolutional Neural Network (CNN) using PyTorch. It is designed to handle image data, performing tasks such as image preprocessing, model training, validation, and performance evaluation. The script is configured to work with datasets organized in a directory structure suitable for `torchvision.datasets.ImageFolder`, typically used for image classification tasks.

### Key Features

- **Data Preprocessing**: Implements transformations including resizing, converting to grayscale, and tensor conversion.
- **Model Definition**: A custom CNN model with two convolutional layers followed by max-pooling, flattening, and fully connected layers.
- **Training and Validation**: Includes a training loop with loss computation, backpropagation, and optimization, along with validation to monitor overfitting.
- **Performance Tracking**: Tracks and prints out training and validation loss for each epoch, and saves the model with the best validation performance.
- **Visualization**: Plots training and validation loss over epochs to visualize learning progress.

### Usage

1. **Data Preparation**: Organize your dataset into respective 'train' and 'validation' folders, each containing subfolders for each class (e.g., 'class1', 'class2').

2. **Configuration**: Set the path to your dataset in the `dataset` variable. Adjust the batch size, number of epochs, and learning rate as needed.

3. **Running the Script**: Execute the script using a Python interpreter compatible with PyTorch. Ensure all dependencies are installed.

### Detailed Description

#### Data Loading and Transformation
- **Transformations**:
  - Images are resized to 64x64 pixels.
  - Converted to grayscale.
  - Transformed into PyTorch tensors.
- **Dataset Split**: The dataset is split into 80% training and 20% validation.

#### Model Architecture
- **Layers**:
  - **Conv1**: First convolutional layer with 32 filters, kernel size of 2, stride of 2.
  - **Pool**: Max pooling layer.
  - **Conv2**: Second convolutional layer with 64 filters, kernel size of 3, stride of 1, padding of 1.
  - **Flatten**: Flattens the output for the dense layer.
  - **FC1**: First fully connected layer with 512 outputs.
  - **FC2**: Output layer with a single neuron for binary classification.
- **Activation**: ReLU activation for hidden layers and Sigmoid for the output layer.

#### Training Process
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss), suitable for binary classification tasks.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: Configurable number of training epochs.
- **Validation**: Computes validation loss to monitor model performance and avoid overfitting.

#### Output
- **Model Saving**: The model with the lowest validation loss is saved for future use.
- **Plotting**: Generates a plot showing training and validation loss over epochs to help visualize the model's learning curve.


## Implementation of RandomForest 

This script, `forest.py`, is designed to classify images using a RandomForest Classifier from the scikit-learn library. The primary application demonstrated in this script is for the classification of pneumonia from chest X-ray images. The script includes processes for image data loading, preprocessing, feature extraction, model training, validation, and evaluation.

### Detailed Script Description
  - The training-validation set is further split to separate training and validation data.

#### RandomForest Model
- **Initialization**:
  - A RandomForest Classifier is initialized with 100 trees and a fixed random state for reproducibility.

- **Training**:
  - The model is trained on the training dataset using pixel values as features.

- **Validation and Evaluation**:
  - The model's performance is evaluated on the validation set using accuracy and ROC-AUC as metrics.
  - Results are printed to provide immediate feedback on performance.

- **Model Saving**:
  - The trained model is saved using joblib for easy reloading and deployment in different environments.
  - 
### Output
- The script prints the model's validation accuracy and ROC-AUC score and saves the RandomForest model to a file named `random_forest_model.joblib`.

### Example Output
```plaintext
Model saved successfully!
Validation Accuracy: 0.9500
Validation ROC-AUC: 0.9875
```

## Running the Scripts

1. **Training the Model**

   The `model_training.py` script facilitates the training process of the RandomForest classifier on the pneumonia dataset. It includes steps for data loading, preprocessing, model training, and saving the trained model for future use.

   ```bash
   python model_training.py
   ```

2. **Evaluating the Model**

   Use the `test.py` script to assess the trained RandomForest model's accuracy on a set of test images. The test data are loaded from `forest.py`, and the script outputs the accuracy score.

   ```bash
   python test.py
   ```

3. **Predicting on New Images**

   The `trial.py` script is designed to apply the trained RandomForest model to classify new chest X-ray images. This script is an example of how the model might be used in a real-world scenario to predict pneumonia.

   ```bash
   python trial.py
   ```
