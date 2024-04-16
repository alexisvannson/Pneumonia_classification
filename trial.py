import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms,datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader



class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0)  # Le 1 indique un seul canal d'entrée (grayscale)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # La sortie de la dernière couche de pooling sera de taille [64, 32, 32] (64 canaux, 32x32 spatial size)
        self.fc1 = nn.Linear(4096, 512)  # Assuming the correct calculation for a 128x128 input image
        self.fc2 = nn.Linear(512, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Utilisez sigmoid à la fin pour la classification binaire
        return x
                        

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resize images to 64x64 pixels
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    #transforms.Normalize(mean=[0.485], std=[0.229])  # #normalized_pixel = (pixel−mean)/std
])

dataset = datasets.ImageFolder(root='/Users/philippevannson/Downloads/chest_xray_2/chest_xray', transform=image_transform)
# Assuming `dataset` is already created with ImageFolder and appropriate transforms
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch size according to your system's capability


# Load the saved model
model = BinaryClassifierCNN() # model initialization
model.load_state_dict(torch.load('./final_model.pth')) # take already prepared weights, final_model.pth, Best_model.pth
model.eval()  # Set the model to evaluation mode
"""
# Load and preprocess the image
image_path = './train/PNEUMONIA/person342_virus_701.jpeg'  # Specify the path to your image
#image_path = './train/NORMAL/IM-0115-0001.jpeg'  # Specify the path to your image

image = Image.open(image_path)
image = image_transform(image)
image = image.unsqueeze(0) # add a batch dimension


with torch.no_grad():
    output = model(image)
    probabilities = output.item()
    
    predicted_class = round(probabilities)

# Define class labels
class_labels = ['NORMAL','PNEUMONIA']

if class_labels[predicted_class] == 'NORMAL':
   probabilities = 1-probabilities
"""

# Print prediction
#print(f'The predicted class is: {class_labels[predicted_class]}, with a probability of {probabilities*100:.2f}%')

correct_predictions = 0
total_images = 0

with torch.no_grad():
    for images, labels in data_loader:
        # Forward pass
        outputs = model(images)
        
        # Convert outputs to binary predictions: 1 if probability > 0.5, else 0
        predicted_classes = (outputs > 0.5).float().squeeze()  # Use .squeeze() to remove any extra dimensions
        
        # Calculate correct predictions
        correct_predictions += (predicted_classes == labels.float()).sum().item()
        total_images += labels.size(0)

# Calculate accuracy
accuracy = correct_predictions / total_images
print(f'Model Accuracy: {accuracy*100:.2f}%')
