import torch
from torch.utils.data import DataLoader

from Dataloader import create_splits
from torchvision import models
import torch.nn as nn
# To calculate precision and recall
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import pandas as pd 

all_labels = []
all_preds = []
required_size = 528  # Use 600 for EfficientNetB7
# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((required_size, required_size)),
    transforms.ToTensor(),
])
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead.")
# Initialize the model
model = models.efficientnet_b6(pretrained=False)  # No need to load pretrained weights

# Adjust the classifier to match the number of classes you trained with
num_classes = 4
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load('best_model_state_dict.pth', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

directory = 'C:/Users/PhongTran/PycharmProjects/RealSenseTest/modified_dataset'

# Create datasets
train_dataset, val_dataset, test_dataset = create_splits(directory, transform, train_size=0, val_size=0, test_size=1)

# Example of creating DataLoaders for each set
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
correct = 0
total = 0
with torch.no_grad():  # No gradients needed as we're just testing
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Move input data to GPU
        labels = labels.to(device)  # Move labels to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
# Calculate precision and recall for each class
precision = precision_score(all_labels, all_preds, average=None)
recall = recall_score(all_labels, all_preds, average=None)
print("Precision per class:", precision)
print("Recall per class:", recall)
# Calculating precision and recall
precision_macro = precision_score(all_labels, all_preds, average='macro')
recall_macro = recall_score(all_labels, all_preds, average='macro')
precision_weighted = precision_score(all_labels, all_preds, average='weighted')
recall_weighted = recall_score(all_labels, all_preds, average='weighted')
precision_micro = precision_score(all_labels, all_preds, average='micro')
recall_micro = recall_score(all_labels, all_preds, average='micro')

print("Macro-average Precision:", precision_macro)
print("Macro-average Recall:", recall_macro)
print("Weighted-average Precision:", precision_weighted)
print("Weighted-average Recall:", recall_weighted)
print("Micro-average Precision:", precision_micro)
print("Micro-average Recall:", recall_micro)
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=[f'Actual Class {i}' for i in range(num_classes)], 
                     columns=[f'Predicted Class {i}' for i in range(num_classes)])

print("Confusion Matrix:")
print(cm_df)
