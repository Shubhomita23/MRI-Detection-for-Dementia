import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import gc


torch.cuda.empty_cache()
gc.collect()


device = torch.device('cpu') 
print(f"Using device: {device}")


print("Loading Alzheimer MRI dataset...")
ds = load_dataset("Falah/Alzheimer_MRI")

print(f"\nDataset splits: {list(ds.keys())}")
print(f"Training samples: {len(ds['train'])}")


first_sample = ds['train'][0]
print(f"Sample keys: {list(first_sample.keys())}")


train_labels = [sample['label'] for sample in ds['train']]
unique_labels, counts = np.unique(train_labels, return_counts=True)

if hasattr(ds['train'].features['label'], 'names'):
    label_names = ds['train'].features['label'].names
    print(f"Label names: {label_names}")
else:
    label_names = [f"Class_{i}" for i in range(len(unique_labels))]
    print(f"Label names not found, using: {label_names}")

num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")
print(f"Label distribution: {dict(zip(unique_labels, counts))}")


sample_image = first_sample['image']
if hasattr(sample_image, 'size'):
    print(f"Image size: {sample_image.size}")
elif hasattr(sample_image, 'shape'):
    print(f"Image shape: {sample_image.shape}")
else:
    img_array = np.array(sample_image)
    print(f"Image shape: {img_array.shape}")


def quick_visualize(dataset, num_samples=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image']
        label = sample['label']
        
        
        if hasattr(image, 'numpy'):
            img_array = image.numpy()
        else:
            img_array = np.array(image)
        
        
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
            img_array = np.transpose(img_array, (1, 2, 0))
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img_array = img_array.squeeze()
        
        axes[i].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
        label_name = label_names[label] if label < len(label_names) else f"Label_{label}"
        axes[i].set_title(f'{label_name} (Label: {label})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

print("\nVisualizing sample images...")
quick_visualize(ds['train'])


class AlzheimerMRIDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.images = []
        self.labels = []
        
        
        print("Pre-loading dataset...")
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image']
            label = sample['label']
            
            
            if hasattr(image, 'convert'):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            
            if self.transform:
                image = self.transform(image)
            else:
                if not isinstance(image, torch.Tensor):
                    image = transforms.ToTensor()(image)
            
            self.images.append(image)
            self.labels.append(label)
        
        print(f"Loaded {len(self.images)} samples")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("\nCreating datasets...")
train_dataset = AlzheimerMRIDataset(ds['train'], transform=transform)


from torch.utils.data import random_split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")

batch_size = 16
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


print("\nTesting data loading...")
for images, labels in train_loader:
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Labels in batch: {labels.tolist()}")
    break


class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes):
        super(AlzheimerCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
      
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = AlzheimerCNN(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_predictions, all_labels
print("\n=== Starting Training ===")
num_epochs = 10
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    print(f'\nEpoch [{epoch+1}/{num_epochs}]')
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_alzheimer_model.pth')
        print(f'  ↳ New best model saved! (Val Acc: {val_acc:.2f}%)')

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
print("\n=== Final Evaluation ===")
model.load_state_dict(torch.load('best_alzheimer_model.pth'))
val_loss, val_acc, predictions, true_labels = validate_epoch(model, val_loader, criterion, device)

print(f"Final Validation Accuracy: {val_acc:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")
print("\n=== Classification Report ===")
print(classification_report(true_labels, predictions, target_names=label_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_names, 
            yticklabels=label_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
print("\n=== Testing on Sample Images ===")
model.eval()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

with torch.no_grad():
    for i in range(6):
        if i >= len(val_subset):
            break
            
        image, true_label = val_subset[i]
        image = image.unsqueeze(0).to(device)
        
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        conf_value = confidence.item()
        img = image.squeeze().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        true_name = label_names[true_label] if true_label < len(label_names) else f"Label_{true_label}"
        pred_name = label_names[predicted_label] if predicted_label < len(label_names) else f"Label_{predicted_label}"
        
        color = 'green' if true_label == predicted_label else 'red'
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name}\nConf: {conf_value:.2f}', color=color)
        axes[i].axis('off')

plt.tight_layout()
plt.show()

print("\n=== Training Complete ===")
print(f"Best model saved as: 'best_alzheimer_model.pth'")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

print("\n=== Sample Predictions ===")
model.eval()
with torch.no_grad():
    for i in range(5):
        image, true_label = val_subset[i]
        image = image.unsqueeze(0).to(device)
        
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        true_name = label_names[true_label]
        pred_name = label_names[predicted.item()]
        conf_value = confidence.item()
        
        status = "✓" if true_label == predicted.item() else "✗"
        print(f"Sample {i+1}: True: {true_name}, Pred: {pred_name}, Confidence: {conf_value:.3f} {status}")