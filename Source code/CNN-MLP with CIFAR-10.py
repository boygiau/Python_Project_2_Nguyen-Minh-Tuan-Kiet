import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from multiprocessing import freeze_support # Import freeze_support

# --- 1. Build MLP Model --- (Moved class definitions outside the main block)
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512) # Layer 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)        # Layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_classes)# Layer 3 (Output)

    def forward(self, x):
        # input_size_mlp needs to be accessible here if not passed.
        # It's better to define it where the model is instantiated or pass it.
        # For now, assuming it's globally defined or passed to __init__ and stored.
        # Let's ensure input_size_mlp is defined before MLP is called.
        x = x.view(-1, 32 * 32 * 3) # Flatten the image
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Build CNN Model --- (Moved class definitions outside the main block)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4) # Flatten for FC layer
        x = self.fc(x)
        return x

# --- 3. Training Function --- (Moved function definitions outside the main block)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cpu'):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# --- 4. Evaluation Function & Plotting --- (Moved function definitions outside)
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.legend(); plt.grid(True)
    plt.suptitle(f'Learning Curves for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def evaluate_model(model, test_loader, model_name, classes_list, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct_test / total_test
    print(f"\n--- {model_name} Test Results ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({correct_test}/{total_test})")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_list, yticklabels=classes_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
    return test_accuracy, avg_test_loss, cm


# This is the main guard
if __name__ == '__main__':
    freeze_support() # Add this line for Windows compatibility with multiprocessing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Preprocess CIFAR-10 ---
    print("\n--- Loading and Preprocessing CIFAR-10 ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_set_full))
    val_size = len(train_set_full) - train_size
    train_set, val_set = random_split(train_set_full, [train_size, val_size])

    batch_size = 64
    # Set num_workers=0 if issues persist, but try with 2 first after the fix
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    input_size_mlp = 32 * 32 * 3

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Number of classes: {num_classes}")

    # --- Hyperparameters ---
    learning_rate = 0.001
    num_epochs = 25

    # --- 6. Train and Evaluate MLP ---
    print("\n\n--- Training and Evaluating MLP ---")
    mlp_model = MLP(input_size=input_size_mlp, num_classes=num_classes).to(device)
    criterion_mlp = nn.CrossEntropyLoss()
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=learning_rate)

    mlp_train_losses, mlp_val_losses, mlp_train_accuracies, mlp_val_accuracies = train_model(
        mlp_model, train_loader, val_loader, criterion_mlp, optimizer_mlp, num_epochs=num_epochs, device=device
    )

    plot_learning_curves(mlp_train_losses, mlp_val_losses, mlp_train_accuracies, mlp_val_accuracies, "MLP")
    mlp_test_acc, mlp_test_loss, mlp_cm = evaluate_model(mlp_model, test_loader, "MLP", classes, device=device)

    # --- 7. Train and Evaluate CNN ---
    print("\n\n--- Training and Evaluating CNN ---")
    cnn_model = CNN(num_classes=num_classes).to(device)
    criterion_cnn = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=learning_rate)

    cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies = train_model(
        cnn_model, train_loader, val_loader, criterion_cnn, optimizer_cnn, num_epochs=num_epochs, device=device
    )

    plot_learning_curves(cnn_train_losses, cnn_val_losses, cnn_train_accuracies, cnn_val_accuracies, "CNN")
    cnn_test_acc, cnn_test_loss, cnn_cm = evaluate_model(cnn_model, test_loader, "CNN", classes, device=device)

    # --- 8. Compare Results ---
    print("\n\n--- Comparison of MLP and CNN ---")
    print(f"MLP Test Accuracy: {mlp_test_acc:.4f}, MLP Test Loss: {mlp_test_loss:.4f}")
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}, CNN Test Loss: {cnn_test_loss:.4f}")
