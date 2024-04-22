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

## Contributing

We welcome contributions to improve the models or extend the project's functionality. Feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is released under the MIT License. For more details, see the LICENSE file in the repository.
