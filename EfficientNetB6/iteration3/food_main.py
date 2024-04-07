import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_dataset_stats(data_loader):
    # Initialize variables to store mean and standard deviation
    mean = torch.zeros(3)  # Assuming RGB images
    std = torch.zeros(3)

    nb_samples = 0

    # Calculate mean and standard deviation
    for data, _ in data_loader:
        # Calculate mean and std per channel
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(dim=2).sum(dim=0)
        std += data.std(dim=2).sum(dim=0)
        nb_samples += batch_samples

    # Calculate overall mean and std
    mean /= nb_samples
    std /= nb_samples

    return mean, std

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_per_process_memory_fraction(0.5)
# Define NN
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GarbageClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes).to(device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.efficientnet(x)

if __name__ == "__main__":
    torch.manual_seed(42)
    temp_loader = DataLoader(ImageFolder("Food101/train", transform=transforms.ToTensor()), batch_size=32, shuffle=True, num_workers=2)

    # Data transformation
    data_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    # Define the path to your Food 101 dataset
    food101_train_path = "Food101/train"
    food101_validation_path = "Food101/validation"

    # Create datasets
    train_dataset = ImageFolder(food101_train_path, transform=data_transform)
    validation_dataset = ImageFolder(food101_validation_path, transform=data_transform)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialization
    num_classes = len(train_dataset.classes)
    model = GarbageClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Training loop
    num_epochs = 15  

    # Initialize variables
    best_val_accuracy = 0.0
    best_model_path = 'best_model.pth'
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0  

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Accumulate total loss
            total_loss += loss.item() * labels.size(0)  

        # Calculate training loss
        train_loss = total_loss / total_samples

        scheduler.step()

        # Validation loop
        model.eval()
        val_outputs, val_labels = [], []
        val_total_loss = 0  

        with torch.no_grad():
            for val_inputs, val_target in validation_loader:
                val_inputs, val_target = val_inputs.to(device), val_target.to(device)
                val_output = model(val_inputs)
                val_loss = criterion(val_output, val_target)

                # Accumulate total validation loss
                val_total_loss += val_loss.item() * val_target.size(0)

                val_outputs.extend(val_output.cpu().numpy())
                val_labels.extend(val_target.cpu().numpy())

        # Calculate validation loss
        val_loss = val_total_loss / len(validation_loader.dataset)

        # Calculate training accuracy
        train_accuracy = total_correct / total_samples

        # Calculate validation accuracy
        val_outputs = torch.tensor(val_outputs)
        val_preds = torch.argmax(val_outputs, dim=1)
        correct = torch.sum(val_preds == torch.tensor(val_labels))
        val_accuracy = correct.item() / len(val_labels)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '  
              f'Training Accuracy: {train_accuracy:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '  
              f'Validation Accuracy: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            print("Saving the model with the best validation loss")
            torch.save(model.state_dict(), best_model_path)
            best_val_loss = val_loss

    # Loading the best model after training
    best_model = GarbageClassifier(num_classes)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    # Calculate training accuracy
    best_model.eval()
    train_outputs, train_labels = [], []
    with torch.no_grad():
        for train_inputs, train_target in train_loader:
            train_inputs, train_target = train_inputs.to(device), train_target.to(device)
            train_outputs.extend(best_model(train_inputs).cpu().numpy())
            train_labels.extend(train_target.cpu().numpy())

    train_outputs = torch.tensor(train_outputs)
    train_preds = torch.argmax(train_outputs, dim=1)
    correct_train = torch.sum(train_preds == torch.tensor(train_labels))
    train_accuracy = correct_train.item() / len(train_labels)

    print(f'Training Accuracy using the best model: {train_accuracy:.4f}')

    # results of validation set using our loaded model
    best_model.eval()
    val_outputs, val_labels = [], []
    with torch.no_grad():
        for val_inputs, val_target in validation_loader:
            val_inputs, val_target = val_inputs.to(device), val_target.to(device)
            val_outputs.extend(best_model(val_inputs).cpu().numpy())
            val_labels.extend(val_target.cpu().numpy())

    val_outputs = torch.tensor(val_outputs)
    val_preds = torch.argmax(val_outputs, dim=1)
    correct_val = torch.sum(val_preds == torch.tensor(val_labels))
    val_accuracy = correct_val.item() / len(val_labels)

    print(f'Validation Accuracy using the best model: {val_accuracy:.4f}')
