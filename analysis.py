import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast  # Corrected import for AMP autocast
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from torchsummary import summary
import torch.nn as nn
from copy import deepcopy

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the modified data
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = resnet20_cifar()

# Load the pre-trained model state_dict
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/save_models/LDNQ_16.pth')
# Checking the data types of weights
for key, tensor in state_dict.items():
    print(f"{key}: {tensor.dtype}")

# Move model to GPU before loading the state_dict
# model.to('cuda')
model.load_state_dict(state_dict)


def evaluate_model(test_loader, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []
    all_top3_probs = []
    all_top3_classes = []
    all_top_probs_by_layer = []

    total_memory_usage = 0  # To keep track of total memory usage during inference

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            labels = labels.to(device)
            input_data = data.to(device).float()  # Start with float32 input data for safety
            
            # print(f'Batch {batch_idx}: Input data shape: {input_data.shape}')
            
            # Use autocast to handle mixed precision automatically
            with autocast('cuda'):  # Corrected AMP usage
                outputs = model(input_data)
                probs = F.softmax(outputs, dim=1)

            max_probs, max_classes = torch.max(probs, 1)
            top_probs, top_classes = torch.topk(probs, 3, dim=1)

            # Calculate memory usage per batch
            batch_memory = input_data.element_size() * input_data.nelement()  # Input data memory
            batch_memory += sum(p.element_size() * p.nelement() for p in model.parameters())  # Parameter memory
            
            total_memory_usage += batch_memory  # Update total memory usage

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            all_top3_probs.extend(top_probs.cpu().numpy() * 100)
            all_top3_classes.extend(top_classes.cpu().numpy())

            for i in range(len(labels)):
                all_top_probs_by_layer.append(probs[i].cpu().numpy())

    print(f'Total memory used during inference: {total_memory_usage / (1024 ** 2):.2f} MB')
    return all_labels, all_preds, all_top3_probs, all_top3_classes, all_top_probs_by_layer



# Free up memory before starting evaluation

torch.cuda.empty_cache()
gc.collect()

# # Proceed with evaluation
actual_labels, predicted_labels, top3_probs, top3_classes, top_probs_by_layer = evaluate_model(test_loader, model)

def calculate_miou(actual_labels, predicted_labels, class_names):
    num_classes = len(class_names)
    # Compute confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(num_classes)))
    
    iou_list = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        union = TP + FP + FN
        if union == 0:
            iou = 0  # Avoid division by zero
        else:
            iou = TP / union
        iou_list.append(iou)
        print(f"Class '{class_names[i]}': IoU = {iou:.4f}")
    
    miou = np.mean(iou_list)
    print(f"Mean IoU (mIoU): {miou:.4f}")
    return miou, iou_list

# Calculate MIOU
num_classes = len(class_names)
miou, iou_list = calculate_miou(actual_labels, predicted_labels, class_names)

# Calculate evaluation metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average='weighted')
recall = recall_score(actual_labels, predicted_labels, average='weighted')
f1 = f1_score(actual_labels, predicted_labels, average='weighted')

# Prepare the data for the Excel file
data = []
for idx, (actual, predicted, probs, classes) in enumerate(zip(actual_labels, predicted_labels, top3_probs, top3_classes)):
    top1_prob, top2_prob, top3_prob = probs
    top1_class, top2_class, top3_class = [class_names[i] for i in classes]

    data.append([
        class_names[actual],
        class_names[predicted],
        top1_prob,
        top1_class,
        top2_prob,
        top2_class,
        top3_prob,
        top3_class,
    ])

# Append the evaluation metrics to the data
data.append(['Metrics', 'Values'])
data.append(['Accuracy', accuracy])
data.append(['Precision', precision])
data.append(['Recall', recall])
data.append(['F1-Score', f1])
# Append MIOU to the data for the Excel file
data.append(['Mean IoU', miou])



# Update the DataFrame and Excel file with MIOU
df = pd.DataFrame(data, columns=[
    'Actual', 'Predicted', 'Top1_Prob', 'Top1_Class', 'Top2_Prob', 'Top2_Class', 'Top3_Prob', 'Top3_Class'
])
df.to_excel('resnet20_onlyMSB.xlsx', index=False)

# Display MIOU
print(f"Mean IoU: {miou}")
print(f"IoU per class: {iou_list}")
print(f"Accuracy : {accuracy}")
print(f"F1 : {f1}")
print(f"Precision {precision}")
print(f"Recall {recall}")


# Generate and save the confusion matrix plot
cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(len(class_names))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix plot
plt.savefig('Confusion_Matrix_resenet20_onlyMSB.png')
plt.show()