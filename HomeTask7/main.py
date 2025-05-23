import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define data transformations
# CIFAR-10 mean and std values
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]

# Normalize transform
norm_transform = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    norm_transform
])

# Validation/Test transforms
val_transform = transforms.Compose([
    transforms.ToTensor(),
    norm_transform
])

# Load CIFAR-10 dataset
# Training set
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)
trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
)

# Validation/Test set
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform
)
testloader = DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
)

# Class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Display some training images
def display_images(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images = images.numpy()

    # Display 6 images
    plt.figure(figsize=(10, 4))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # Transpose to HWC format for matplotlib
        img = np.transpose(images[i], (1, 2, 0))
        # Un-normalize
        img = img * np.array(cifar10_std) + np.array(cifar10_mean)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"{classes[labels[i]]}")
    plt.tight_layout()
    plt.show(block=True)

# display_images(trainloader)
# Define model based on the diagram
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Single ReLU activation for all layers
        self.relu = nn.ReLU()

        # First convolutional layer (5x5, stride 1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        # Average pooling layer (2x2)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second convolutional layer (5x5, stride 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1)
        # Average pooling layer (2x2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool2(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


# Create the model
model = SimpleCNN().to(device)
print(model)
summary(model, (3, 32, 32))

# Print model parameters
num_params = sum(param.numel() for param in model.parameters())
num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Total parameters: {num_params}")
print(f"Trainable parameters: {num_trainable_params}")

# Set up training parameters
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Set up TensorBoard writer
tboard_writer = SummaryWriter('runs/CIFAR10-CNN')

# Training loop
epochs = 50
train_losses = []
val_losses = []
best_accuracy = 0

for epoch in range(epochs):
    # Training phase
    model.train()

    all_Y_train_epoch = np.array([]).reshape(0, 1)
    all_Yhat_train_epoch = np.array([]).reshape(0, 1)
    all_train_losses_epoch = np.array([])

    for X_train, Y_train in trainloader:
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        # Forward pass
        logits = model(X_train)

        # Compute loss
        loss = loss_fn(logits, Y_train)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store metrics for current batch
        y_hat = F.softmax(logits, dim=1)
        y_hat = y_hat.detach().cpu().numpy()
        y_hat = np.argmax(y_hat, axis=1)
        y_hat = y_hat.reshape(-1, 1)

        Y_train_np = Y_train.detach().cpu().numpy()
        Y_train_np = Y_train_np.reshape(-1, 1)

        all_Y_train_epoch = np.vstack((all_Y_train_epoch, Y_train_np))
        all_Yhat_train_epoch = np.vstack((all_Yhat_train_epoch, y_hat))
        all_train_losses_epoch = np.append(all_train_losses_epoch, loss.item())

    # Compute metrics for current training epoch
    train_losses.append(all_train_losses_epoch.mean())
    ac_train = accuracy_score(all_Y_train_epoch, all_Yhat_train_epoch)
    cm_train = confusion_matrix(all_Y_train_epoch, all_Yhat_train_epoch)

    # Validation phase
    model.eval()

    all_Y_val_epoch = np.array([]).reshape(0, 1)
    all_Yhat_val_epoch = np.array([]).reshape(0, 1)
    all_val_losses_epoch = np.array([])

    with torch.no_grad():
        for X_val, Y_val in testloader:
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)

            # Forward pass
            logits = model(X_val)

            # Compute loss
            loss = loss_fn(logits, Y_val)

            # Store metrics for current batch
            y_hat_val = F.softmax(logits, dim=1)
            y_hat_val = y_hat_val.detach().cpu().numpy()
            y_hat_val = np.argmax(y_hat_val, axis=1)
            y_hat_val = y_hat_val.reshape(-1, 1)

            Y_val_np = Y_val.detach().cpu().numpy()
            Y_val_np = Y_val_np.reshape(-1, 1)

            all_Y_val_epoch = np.vstack((all_Y_val_epoch, Y_val_np))
            all_Yhat_val_epoch = np.vstack((all_Yhat_val_epoch, y_hat_val))
            all_val_losses_epoch = np.append(all_val_losses_epoch, loss.item())

    # Compute metrics for current validation epoch
    val_losses.append(all_val_losses_epoch.mean())
    ac_val = accuracy_score(all_Y_val_epoch, all_Yhat_val_epoch)
    cm_val = confusion_matrix(all_Y_val_epoch, all_Yhat_val_epoch)

    print(
        f"Epoch {epoch + 1}/{epochs}, Train Acc: {ac_train:.4f}, Val Acc: {ac_val:.4f}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Save checkpoint if better accuracy
    if ac_val > best_accuracy:
        best_accuracy = ac_val
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses,
            'val_loss': val_losses,
            'train_acc': ac_train,
            'val_acc': ac_val,
            'confusion_matrix': cm_val
        }
        torch.save(checkpoint, 'cifar10_best.pth')
        print(f"Checkpoint saved with validation accuracy: {best_accuracy:.4f}")

    # Log metrics to TensorBoard
    tboard_writer.add_scalar("Loss/train", train_losses[-1], epoch)
    tboard_writer.add_scalar("Loss/val", val_losses[-1], epoch)
    tboard_writer.add_scalar("Accuracy/train", ac_train, epoch)
    tboard_writer.add_scalar("Accuracy/val", ac_val, epoch)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('cifar10_loss.png')
plt.show(block=True)

# Load best model
print("\nLoading best model...")
checkpoint = torch.load('cifar10_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
best_epoch = checkpoint['epoch']
best_val_acc = checkpoint['val_acc']
best_train_acc = checkpoint['train_acc']
confusion_mat = checkpoint['confusion_matrix']

print(f"Best model from epoch {best_epoch + 1} with:")
print(f"  Training accuracy: {best_train_acc:.4f}")
print(f"  Validation accuracy: {best_val_acc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Add text annotations in the confusion matrix
thresh = confusion_mat.max() / 2.
for i in range(confusion_mat.shape[0]):
    for j in range(confusion_mat.shape[1]):
        plt.text(j, i, format(confusion_mat[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('cifar10_confusion_matrix.png')
plt.show(block=True)


# Test on a few examples
def test_model_on_examples(model, dataloader, num_examples=6):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # Display results
    plt.figure(figsize=(12, 6))
    for i in range(num_examples):
        plt.subplot(2, 3, i + 1)
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        img = img * np.array(cifar10_std) + np.array(cifar10_mean)
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        predicted_label = classes[predicted[i].item()]
        true_label = classes[labels[i].item()]
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
        if predicted_label == true_label:
            plt.gca().spines['bottom'].set_color('green')
            plt.gca().spines['top'].set_color('green')
            plt.gca().spines['right'].set_color('green')
            plt.gca().spines['left'].set_color('green')
        else:
            plt.gca().spines['bottom'].set_color('red')
            plt.gca().spines['top'].set_color('red')
            plt.gca().spines['right'].set_color('red')
            plt.gca().spines['left'].set_color('red')
    plt.tight_layout()
    plt.savefig('cifar10_test_examples.png')
    plt.show(block=True)


# Test model on examples
test_model_on_examples(model, testloader)