import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Assuming get_dataloader is defined and imported
# from your_data_loading_module import get_dataloader

# Load the test data using the provided function
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = resnet20_cifar()
model.load_state_dict(torch.load('/scratch/users/Maryam/L-DNQ/ResNet20/save_models/LDNQ.pth'))
model.eval()

# Function to perform inference and collect actual vs predicted labels
def evaluate_model(test_loader, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Perform inference
actual_labels, predicted_labels = evaluate_model(test_loader, model)

# Save actual vs predicted labels to a text file
with open('actual_vs_predicted_LDNQ.txt', 'w') as f:
    f.write('Actual\tPredicted\n')
    for actual, predicted in zip(actual_labels, predicted_labels):
        f.write(f'{actual}\t{predicted}\n')

# Create a confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels, labels=range(len(class_names)))

# Save the confusion matrix to a text file
with open('confusion_matrix_LDNQ.txt', 'w') as f:
    f.write('Confusion Matrix\n')
    f.write('\t' + '\t'.join(class_names) + '\n')
    for i, row in enumerate(cm):
        f.write(class_names[i] + '\t' + '\t'.join(map(str, row)) + '\n')

print("Results saved to actual_vs_predicted.txt and confusion_matrix.txt")
