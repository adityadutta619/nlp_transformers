import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for preprocessing the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=r'E:\dl_data\mnist', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=r'E:\dl_data\mnist', train=False, transform=transform, download=True)

# DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Fixed input size
        self.fc2 = nn.Linear(128, 10)  # Output layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        #print(f"Shape after conv2 and pooling: {x.shape}")  # Debugging step
        x = torch.flatten(x, 1)  # Flatten before FC layers
        #print(f"Shape after flattening: {x.shape}")  # Debugging step
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model
model = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        """ import matplotlib.pyplot as plt
        import torchvision

        # Get one sample image and label from the dataset
        sample_image, label = train_dataset[0]  # Get first image from training set

        # Convert the image tensor to a NumPy array
        sample_image = sample_image.squeeze().numpy()  # Remove channel dimension (1, 28, 28) → (28, 28)

        # Display the image
        plt.imshow(sample_image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis("off")  # Hide axes
        plt.show()
        """
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
