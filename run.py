from Dataloader import CustomDataset, create_splits
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter




# Example usage
directory = 'C:/Users/PhongTran/PycharmProjects/RealSenseTest/modified_dataset'

required_size = 528  # Use 600 for EfficientNetB7
# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((required_size, required_size)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset, val_dataset, test_dataset = create_splits(directory, transform)

# Example of creating DataLoaders for each set
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained EfficientNet model
model = models.efficientnet_b6(pretrained=True)  # You can change to efficientnet_b7

# Freeze the parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Change the input layer if necessary (optional, here for demonstration)
# EfficientNet models in PyTorch adjust automatically to different input sizes,
# but if you need a specific input processing layer, you might need to adjust or add it manually

# Modify the output layer to fit the number of classes you need
num_classes = 4  # Change this to your number of classes
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

# Use the appropriate loss function for your classification task
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

writer = SummaryWriter('runs/experiment_name')
best_val_loss = float('inf')
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead.")
model = model.to(device)

# Training loop (skeleton)
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_val_loss = 0
    total_correct = 0
    total_images = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # Move input data to GPU
        labels = labels.to(device)  # Move labels to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * inputs.size(0)

    # Log training loss
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)  # Move input data to GPU
            labels = labels.to(device)  # Move labels to GPUoutputs = model(inputs)
            val_loss = criterion(outputs, labels)
            # calculate accuracy etc.
            total_val_loss += val_loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)
    # Log validation loss and accuracy
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    accuracy = 100 * total_correct / total_images
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', accuracy, epoch)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}, Accuracy: {accuracy:.2f}%")
        # Check if the current model is the best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_state_dict.pth')
        print(f'New best model saved at epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f}')

writer.close()
print("Training and validation complete.")
