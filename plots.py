import matplotlib.pyplot as plt

# Learning rates
learning_rates = [0.0001, 0.001, 0.01]

# Training accuracies
train_accuracies_resnet50 = [91.61, 86.44, 30.01]
train_accuracies_resnet101 = [94.26, 87.55, 24.59]

# Validation accuracies
val_accuracies_resnet50 = [83.07, 78.04, 28.93]
val_accuracies_resnet101 = [84, 78.87, 23.54]

# Plotting training accuracies
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, train_accuracies_resnet50, marker='o', linestyle='-', color='r', label='ResNet-50 Training Accuracy')
plt.plot(learning_rates, train_accuracies_resnet101, marker='o', linestyle='-', color='b', label='ResNet-101 Training Accuracy')

# Plotting validation accuracies
plt.plot(learning_rates, val_accuracies_resnet50, marker='o', linestyle='--', color='r', label='ResNet-50 Validation Accuracy')
plt.plot(learning_rates, val_accuracies_resnet101, marker='o', linestyle='--', color='b', label='ResNet-101 Validation Accuracy')

plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Learning Rate')
plt.legend()
plt.grid(True)
plt.show()

# Weight decays
weight_decays = [1e-5, 1e-4, 1e-3]

# Training accuracies
train_accuracies_resnet50_wd = [91.61, 91.36, 90.64]
train_accuracies_resnet101_wd = [94.26, 94.17, 93.31]

# Validation accuracies
val_accuracies_resnet50_wd = [83.07, 82.73, 82.91]
val_accuracies_resnet101_wd = [84, 83.82, 84.02]

# Plotting training accuracies
plt.figure(figsize=(10, 6))
plt.plot(weight_decays, train_accuracies_resnet50_wd, marker='o', linestyle='-', color='r', label='ResNet-50 Training Accuracy')
plt.plot(weight_decays, train_accuracies_resnet101_wd, marker='o', linestyle='-', color='b', label='ResNet-101 Training Accuracy')

# Plotting validation accuracies
plt.plot(weight_decays, val_accuracies_resnet50_wd, marker='o', linestyle='--', color='r', label='ResNet-50 Validation Accuracy')
plt.plot(weight_decays, val_accuracies_resnet101_wd, marker='o', linestyle='--', color='b', label='ResNet-101 Validation Accuracy')

plt.xscale('log')
plt.xlabel('Weight Decay')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Weight Decay')
plt.legend()
plt.grid(True)
plt.show()