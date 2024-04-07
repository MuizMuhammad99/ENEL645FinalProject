import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
import torch.backends.cudnn as cudnn
from tempfile import TemporaryDirectory
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import os
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.datasets import ImageFolder

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
# Data Directory in TALC
data_dir = 'food-101'

#2. Pre-process Data
# Data Transformations
data_transforms = {
    # Data Augmentation
    'train': transforms.Compose([
        # Crops image
        transforms.RandomResizedCrop(224),
        # Flips Image
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Converts Image to a Tensor
        transforms.ToTensor(),
        # Normalizes Image
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Data Normalization
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 3. Experimental Setup with Train, Validation, Test Split

# Define the path to your Food 101 dataset
food101_train_path = "food-101/train"
food101_validation_path = "food-101/validation"

# Create datasets
train_dataset = ImageFolder(food101_train_path, data_transforms['train'])
validation_dataset = ImageFolder(food101_validation_path, data_transforms['validation'])

# Creating Dataloaders for the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=2)

# Finding out the Dataset sizes per dataset
train_sizes = len(train_dataset)
val_sizes = len(validation_dataset)
# Finding the Class names
class_names = train_dataset.classes

# Printing dataset info
print("Train set:", train_sizes)
print("Val set:", val_sizes)
print("Class names:", class_names)

# Creating iterators and batch for training data
train_iter = iter(train_loader)
train_batch = next(train_iter)
print(train_batch[0].size())
print(train_batch[1].size())

# 4. Transfer Learning and Convolution
class FoodClassifier(nn.Module):
    def __init__(self, num_class, input_shape, transfer=False):
        """
        Initialize the Food Model class.

        Args:
            num_class (int): Number of output classes.
            input_shape (tuple): Shape of input data (channels, height, width).
            transfer (bool): Whether to use transfer learning with a pretrained ResNet18.
        """
        super().__init__()

        # Initializing the parameters
        self.num_class = num_class
        self.input_shape = input_shape
        self.transfer = transfer

        # Initializing the feature Extractor (ResNet18)
        self.feature_extract = models.resnet18(weights=transfer)
        
        # Transfer learing
        if self.transfer:
            # Freezing layers
            self.feature_extract.eval()
            # Freezing Parameters
            for param in self.feature_extract.parameters():
                param.requires_grad = False
        
        # Number of features for classifier input
        n_features = self.get_conv_output(self.input_shape)
        
        # Classifier model to classify features into each class (fully connected layer)
        self.classifier = nn.Linear(n_features, num_class)

    def get_conv_output(self, shape):
        """
        Calculate the number of features in the output of the convolutional layers.

        Args:
            shape (tuple): Shape of the input data (channels, height, width).

        Returns:
            int: Number of features in the output of the convolutional layers.
        """
        batch_size = 1
        temp_in = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feature = self.feature_extract(temp_in)
        n_size = output_feature.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the network.
        """
        x = self.feature_extract(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

num_classes = len(class_names)
net = FoodClassifier(num_classes, (3,224,224), True)
net.to(device)

# 5. Loss and Metrics, Experiment Tracking
# Fine Tuned Model for training
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    """
    Train the given model.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion: Loss function.
        optimizer: Optimizer for parameter optimization.
        scheduler: Learning rate scheduler.
        num_epochs (int): Number of epochs for training.

    Returns:
        torch.nn.Module: Trained model.
    """

    # Creating a path to save the best model
    PATH = 'bestfoodpath_resnet18_0.0001lr.pth'
    # best loss function
    best_loss = 1e+20

    # loops over the dataset epoch number of times
    for epoch in range(num_epochs):
        # Training Loop
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Inputs
            inputs, labels = data[0].to(device), data[1].to(device)
            # Zeros the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
        scheduler.step()

        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                # Inputs
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            print(f'validation loss: {val_loss / i:.3f}')

            # Saving the best model
            if val_loss < best_loss:
                print("Saving model")
                torch.save(model.state_dict(), PATH)
                best_loss = val_loss
        

    print('Finished Training')
    return model

# 6. Hyperparameters

# Loss Function
criterion = nn.CrossEntropyLoss()
# Parameter Optimization
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.0001)
# Scheduler
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)


# 7. Training the food model
#net = train_model(net, criterion, optimizer, scheduler,
#                       num_epochs=15)

# Load the best model to be used in the validation set
net = FoodClassifier(num_classes, (3,224,224), True)
net.load_state_dict(torch.load('bestfoodpath_resnet18_0.0001lr.pth'))
net.to(device)

# 8. Calculate training accuracy
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(train_loader, 0):
        # calculate outputs by running images through the network
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        # the class  with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n\nAccuracy on the test images: {100 * correct / total} %')

# 9. Testing the food model
# Final Accuracy of the model
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(val_loader, 0):
        # calculate outputs by running images through the network
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        # the class  with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 10. Final Accuracy score, Class accuracy score and confusion matrix
print(f'Accuracy on the test images: {100 * correct / total} %\n\n')

# Class Accuracy Variables
class_correct = list(0. for _ in range(len(class_names)))
class_total = list(0. for _ in range(len(class_names)))

with torch.no_grad():
    for data in val_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        # Update class accuracy
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print class accuracy
for i in range(len(class_names)):
    print(f'Accuracy of {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
