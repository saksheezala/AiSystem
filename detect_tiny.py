# detect_tiny.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from utils_tiny import load_tiny_imagenet
from train_tiny import load_model
from adversarial_tiny import generate_fgsm_examples

class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        # Third conv block (added for increased capacity)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        # Fully connected layers
        self.fc1   = nn.Linear(128 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 1)  # Predict continuous epsilon value

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Regression output: predicted epsilon
        return x

def train_regression_detector(num_epochs=30, batch_size=64, epsilons=[0.1, 0.2, 0.3, 0.4, 0.5], num_batches=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load_tiny_imagenet(batch_size=batch_size, train=True)
    # Load the pre-trained TinyNet used to generate adversarial examples
    cnn_model = load_model().to(device)
    
    images_list, epsilon_list = [], []
    
    # Generate normal images (no attack, so epsilon=0)
    normal_imgs, _ = generate_fgsm_examples(cnn_model, device, data_loader, epsilon=0, num_batches=num_batches)
    images_list.append(normal_imgs)
    epsilon_list.append(torch.zeros(normal_imgs.size(0), dtype=torch.float32))
    
    # Generate adversarial images for each specified epsilon value
    for eps in epsilons:
        _, adv_imgs = generate_fgsm_examples(cnn_model, device, data_loader, epsilon=eps, num_batches=num_batches)
        images_list.append(adv_imgs)
        epsilon_list.append(torch.full((adv_imgs.size(0),), eps, dtype=torch.float32))
    
    X = torch.cat(images_list, dim=0)
    y = torch.cat(epsilon_list, dim=0).unsqueeze(1)  # Shape: (N, 1)
    
    dataset = data.TensorDataset(X, y)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    regression_model = RegressionCNN().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(regression_model.parameters(), lr=0.0005)
    
    print("Starting training of the regression detector...")
    for epoch in range(num_epochs):
        regression_model.train()
        running_loss = 0.0
        total_samples = 0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = regression_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
    torch.save(regression_model.state_dict(), "regression_detector_tiny.pth")
    print("Regression detector saved as regression_detector_tiny.pth")
    return regression_model

if __name__ == "__main__":
    train_regression_detector()
