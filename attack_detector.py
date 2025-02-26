# attack_detector.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from utils_tiny import load_tiny_imagenet
from train_tiny import load_model
from adversarial_tiny import generate_fgsm_examples
from pgd_tiny import generate_pgd_examples

class AttackClassifier(nn.Module):
    def __init__(self):
        super(AttackClassifier, self).__init__()
        # First convolutional block: Input 3x64x64, Output 32x64x64 then pool to 32x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block: Output 64x32x32 then pool to 64x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block: Output 128x16x16 then pool to 128x8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 3)  # 3 classes: 0-Normal, 1-FGSM, 2-PGD

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_attack_classifier(num_epochs=30, batch_size=64, num_batches=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Tiny ImageNet dataset
    data_loader = load_tiny_imagenet(batch_size=batch_size, train=True)
    # Load pre-trained TinyNet used to generate adversarial examples
    cnn_model = load_model().to(device)
    
    images_list = []
    labels_list = []
    
    # 1. Normal Images (Label 0): Generate using FGSM with epsilon = 0 (i.e. no attack)
    normal_imgs, _ = generate_fgsm_examples(cnn_model, device, data_loader, epsilon=0, num_batches=num_batches)
    images_list.append(normal_imgs)
    labels_list.append(torch.zeros(normal_imgs.size(0), dtype=torch.long))
    
    # 2. FGSM Images (Label 1): Generate FGSM adversarial examples (e.g., using epsilon=0.3)
    _, fgsm_imgs = generate_fgsm_examples(cnn_model, device, data_loader, epsilon=0.3, num_batches=num_batches)
    images_list.append(fgsm_imgs)
    labels_list.append(torch.ones(fgsm_imgs.size(0), dtype=torch.long))
    
    # 3. PGD Images (Label 2): Generate PGD adversarial examples (e.g., epsilon=0.3, num_iter=40, alpha=0.01)
    _, pgd_imgs = generate_pgd_examples(cnn_model, device, data_loader, epsilon=0.3, num_iter=40, alpha=0.01, num_batches=num_batches)
    images_list.append(pgd_imgs)
    labels_list.append(torch.full((pgd_imgs.size(0),), 2, dtype=torch.long))
    
    # Concatenate the datasets
    X = torch.cat(images_list, dim=0)
    y = torch.cat(labels_list, dim=0)
    
    dataset = data.TensorDataset(X, y)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    model = AttackClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training of attack classifier...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / total
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    torch.save(model.state_dict(), "attack_classifier_tiny.pth")
    print("Attack classifier saved as attack_classifier_tiny.pth")
    return model

if __name__ == "__main__":
    train_attack_classifier()
