import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# Define transformations
transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resize images to 64x64 pixels
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.ToTensor(),        # Convert images to PyTorch tensors
    #transforms.Normalize(mean=[0.485], std=[0.229])  # #normalized_pixel = (pixel−mean)/std
])

# Load the dataset from the file system
dataset = datasets.ImageFolder(root='fruits-metadata.json', transform=transform)

# Optionally, split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Create DataLoader instances for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)


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

# Initialiser le modèle
model = BinaryClassifierCNN()

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()  # BCELoss est adapté pour la classification binaire
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_errors = []
val_errors = []
num_epochs = 15
best_val_loss = float('inf')  # Initialize the best validation loss to infinity
patience = 2  # How many epochs to wait after last time validation loss improved.
patience_counter = 0  # Counter for patience

for epoch in range(num_epochs):
    model.train()  
    total_train_loss = 0.0
    for inputs, labels in train_loader:  # Assuming you have a DataLoader named train_loader
        # Forward pass
        y_pred = model.forward(inputs)
        y_pred = y_pred.squeeze() 
        labels = labels.float()  
        loss = criterion(y_pred,labels) 
        total_train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()  
      # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    train_errors.append(avg_train_loss)

 
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():  # disable gradient computation (more efficient and reduces memory usage)
        for images, labels in valid_loader:
            outputs = model(images)
            labels = labels.float()  # Ensure labels are floating-point for BCELoss

            # There's no need to squeeze the outputs as your last layer is a sigmoid, expected to work with BCELoss directly.
            # Just ensure that labels and outputs are of the same dimension.
            loss = criterion(outputs.squeeze(), labels)  # Compute the loss, ensuring output dimensions match labels

            total_val_loss += loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = total_val_loss / len(valid_loader)
    val_errors.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    # Save model if validation loss has decreased
    if avg_val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...')
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'Best_model.pth')
        patience_counter = 0  # Reset patience counter after improvement


# Plotting the training and validation errors
epochs_range = range(1, len(train_errors) + 1)

plt.plot(epochs_range, train_errors, label='Training Error')
plt.plot(epochs_range, val_errors, label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()