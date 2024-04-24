import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from joblib import dump

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resize images to 64x64 pixels
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load the dataset from the file system
dataset = datasets.ImageFolder(root='/Users/philippevannson/Desktop/code/pneumonia_classification/train', transform=transform)

# Split the dataset into features and labels
def extract_features_and_labels(dataset):
    features = []
    labels = []
    for img, label in dataset:
        features.append(img.flatten().numpy())  # Flatten the 2D image into 1D and convert to numpy
        labels.append(label)
    return np.array(features), np.array(labels)

features, labels = extract_features_and_labels(dataset)

# Split data into training + validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Further split the training + validation set into separate training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_valid)
probabilities = rf.predict_proba(X_valid)[:, 1]  # Assuming positive class is labeled as '1'

dump(rf, 'random_forest_model.joblib')

print("Model saved successfully!")


# Evaluate the model
accuracy = accuracy_score(y_valid, predictions)
roc_auc = roc_auc_score(y_valid, probabilities)

print(f'Validation Accuracy: {accuracy:.4f}')
print(f'Validation ROC-AUC: {roc_auc:.4f}')
