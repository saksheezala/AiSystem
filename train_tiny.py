# train_tiny.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils_tiny import load_tiny_imagenet

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        # For 64x64 images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64->32
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32->16
        
        self.dropout = nn.Dropout(0.5)
        # After two poolings, 64->32->16. With conv layers, we assume final spatial size 16x16.
        # To further reduce, you might add another pooling. Here, let's add one more pooling:
        self.pool3 = nn.MaxPool2d(2, 2)  # 16->8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 200)  # Tiny ImageNet has 200 classes
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (64, 32, 32)
        x = F.relu(self.bn2(self.conv2(x)))              # (128, 32, 32)
        x = self.pool1(x)                                # (128, 16, 16)
        
        x = F.relu(self.bn3(self.conv3(x)))              # (256, 16, 16)
        x = F.relu(self.bn4(self.conv4(x)))              # (256, 16, 16)
        x = self.pool2(x)                                # (256, 8, 8)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(num_epochs=10, batch_size=128, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_tiny_imagenet(batch_size=batch_size, train=True)
    model = TinyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_samples = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        total_samples += total
        
    torch.save(model.state_dict(), "tiny_net.pth")
    print("TinyNet model saved as tiny_net.pth")
    return model

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyNet().to(device)
    model.load_state_dict(torch.load("tiny_net.pth", map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    train_model()
