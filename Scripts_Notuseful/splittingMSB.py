import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from zennit.torchvision import ResNetCanonizer
from zennit.core import Composite
from zennit.rules import Epsilon
from zennit.canonizers import Canonizer
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm


# Install a library for LRP if you don't have it, for example, `innvestigate` or `zennit`
# This code assumes LRP implementation using a library like 'innvestigate' or 'zennit'.

model = resnet20_cifar()

# Load the pre-trained model state_dict
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

try:
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {state_dict}.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# model.eval()

# 2. Perform Layer-wise Relevance Propagation (LRP)
# Properly instantiate the canonizers
resnet_canonizer = ResNetCanonizer()  # Instantiate the ResNetCanonizer
batch_norm_canonizer = SequentialMergeBatchNorm()  # Instantiate the BatchNorm canonizer

# Print debug information
print(f"ResNetCanonizer type: {type(resnet_canonizer)}")
print(f"BatchNorm Canonizer type: {type(batch_norm_canonizer)}")

# Define a Composite with Epsilon rule and properly instantiated canonizers
composite = Composite([Epsilon()], canonizers=[resnet_canonizer, batch_norm_canonizer])  # Correct setup

# 3. Perform LRP on a sample input to get relevance scores
def compute_lrp(model, input_tensor, composite):
    """
    Compute LRP relevance scores for each layer of the model.
    
    Args:
        model (nn.Module): The model to analyze.
        input_tensor (torch.Tensor): Input tensor to compute relevance for.
        composite (Composite): Zennit composite object defining LRP rules.
    
    Returns:
        dict: A dictionary containing relevance scores for each layer.
    """
    relevance_scores = {}

    # Use a context manager to apply the composite during forward and backward passes
    print("Entering composite context manager for LRP computation...")
    with composite.context(model) as lrp:
        print("Composite context manager entered successfully.")
        output = model(input_tensor)
        target_class = output.argmax(dim=1)  # Get the class with the highest score
        print(f"Model output computed. Target class: {target_class.item()}")
        
        relevance = lrp(input_tensor, target_class)
        print("Relevance computed successfully.")

        # Gather relevance scores layer by layer
        for name, layer in model.named_children():
            if hasattr(layer, 'weight'):
                relevance_scores[name] = relevance[layer.weight].detach().cpu().numpy()
                print(f"Relevance for layer {name} computed and stored.")
    
    return relevance_scores

# 4. Prepare the Input Tensor
input_tensor = torch.randn((1, 3, 32, 32))  # Example input for CIFAR-10

# 5. Compute LRP Relevance Scores
print("Starting LRP computation...")
try:
    relevance_scores = compute_lrp(model, input_tensor, composite)
    print("LRP Relevance Scores Computed.")
except Exception as e:
    print(f"Error during LRP computation: {e}")

# 3. Analyze LRP Results to Determine Sensitivity
print(f"Relevance Scores: {relevance_scores}")
relevance_scores = np.array(relevance_scores)
normalized_scores = relevance_scores / np.max(relevance_scores)
print(f"Normalized Relevance Scores: {normalized_scores}")

# Set a threshold and classify layers
sensitivity_threshold = 0.7  # Example threshold; tune this based on the distribution
critical_layers = []
less_critical_layers = []

for i, score in enumerate(normalized_scores):
    if score >= sensitivity_threshold:
        critical_layers.append(i)
        print(f"Layer {i} classified as critical.")
    else:
        less_critical_layers.append(i)
        print(f"Layer {i} classified as less critical.")

assert len(critical_layers) > 0, "No critical layers identified; check LRP scores or threshold."
assert len(less_critical_layers) > 0, "No less critical layers identified; check LRP scores or threshold."

# MSB Manipulation Function with dtype Conversion
def truncate_to_msb_and_convert(tensor, num_bits=16, dtype=torch.float16):
    """
    Truncate the tensor to keep only the most significant bits (MSB) and convert to a lower precision data type.
    
    Args:
        tensor (torch.Tensor): Input tensor to truncate and convert.
        num_bits (int): Number of bits to retain in MSB.
        dtype (torch.dtype): Target data type for conversion.
    
    Returns:
        torch.Tensor: Tensor with MSB retained and converted to lower precision.
    """
    # Ensure the tensor is float32 for bit manipulation
    tensor = tensor.float()
    # Calculate the scaling factor to keep only MSB
    factor = 2 ** (32 - num_bits)
    # Retain only the MSB
    truncated_tensor = torch.round(tensor / factor) * factor
    # Convert the tensor to a lower precision data type
    converted_tensor = truncated_tensor.to(dtype)
    return converted_tensor

# 2. Apply MSB and Convert to Less Critical Layers
def apply_msb_and_convert_to_model(model, less_critical_layers, num_bits=16, dtype=torch.float16):
    """
    Apply MSB truncation and convert to a lower precision data type for less critical layers of the model.
    
    Args:
        model (nn.Module): The model to modify.
        less_critical_layers (list): List of layer indices considered less critical.
        num_bits (int): Number of bits to retain in MSB.
        dtype (torch.dtype): Target data type for conversion.
    
    Returns:
        nn.Module: Modified model with MSB applied and converted to less critical layers.
    """
    for i, layer in enumerate(model.children()):
        if i in less_critical_layers:
            print(f"Applying MSB truncation and converting layer {i} to {dtype}.")
            # Check if layer has parameters (weights)
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data = truncate_to_msb_and_convert(layer.weight.data, num_bits, dtype)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = truncate_to_msb_and_convert(layer.bias.data, num_bits, dtype)
    return model

# Apply MSB truncation and conversion to less critical layers
model_with_msb = apply_msb_and_convert_to_model(model, less_critical_layers, num_bits=16, dtype=torch.float16)
print("MSB truncation and conversion to lower precision applied to less critical layers.")


# # 4. Modify the Model for Mixed Precision
# def convert_to_mixed_precision(model, critical_layers, less_critical_layers):
#     print("Converting model to mixed precision...")
#     for i, layer in enumerate(model.children()):
#         if i in critical_layers:
#             print(f"Layer {i} set to 32-bit precision.")
#             layer.float()  # Use full 32-bit precision
#         elif i in less_critical_layers:
#             print(f"Layer {i} set to 16-bit precision.")
#             layer.half()  # Use reduced 16-bit precision
#     return model

# # Convert the model
# model = convert_to_mixed_precision(model, critical_layers, less_critical_layers)
# print("Model conversion to mixed precision completed.")

# Save the modified model
modified_model_path = '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/SmallerModels/modified_model.pth' 
try:
    torch.save(model.state_dict(), modified_model_path)
    print(f"Modified model saved successfully at {modified_model_path}.")
except Exception as e:
    print(f"Error saving the modified model: {e}")
    raise


# Load the modified data
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 5. Validate the Modified Model
def validate_model(model, test_loader, device='cuda'):
    print("Starting model validation...")
    model.eval()
    model.to(device)
    total, correct = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            try:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during validation at batch {batch_idx}: {e}")
                continue

            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"Batch {batch_idx}: Current Accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy

# Assuming you have a DataLoader for CIFAR-10, replace with your data loader
# validate_model(model, test_loader)

# Function to perform inference and collect actual vs predicted labels and top 3 probabilities
def evaluate_model(test_loader, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Check model precision and ensure all layers are consistent
    model = model.to(torch.float16) if next(model.parameters()).dtype == torch.float16 else model.to(torch.float32)
    
    all_preds = []
    all_labels = []
    all_top3_probs = []
    all_top3_classes = []
    all_top_probs_by_layer = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            labels = labels.to(device)
            input_data = data.to(device)

            # Convert input data to match model precision
            if next(model.parameters()).dtype == torch.float16:
                input_data = input_data.half()
            else:
                input_data = input_data.float()

            print(f'Batch {batch_idx}: Input data shape: {input_data.shape}')
            
            # Forward pass
            outputs = model(input_data)
            probs = F.softmax(outputs, dim=1)
            max_probs, max_classes = torch.max(probs, 1)

            # Debugging: print top 3 probabilities and their classes
            top_probs, top_classes = torch.topk(probs, 3, dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get top 3 probabilities and their corresponding classes for each example
            all_top3_probs.extend(top_probs.cpu().numpy() * 100)  # Convert to percentage
            all_top3_classes.extend(top_classes.cpu().numpy())

            # Ensure each example has probabilities for 3 layers, filling with NaNs if necessary
            for i in range(len(labels)):
                all_top_probs_by_layer.append(probs[i].cpu().numpy())

    return all_labels, all_preds, all_top3_probs, all_top3_classes, all_top_probs_by_layer

# Perform inference
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
df.to_excel('onlyMSB.xlsx', index=False)

# Display MIOU
print(f"Mean IoU: {miou}")
print(f"IoU per class: {iou_list}")

# Generate and save the confusion matrix plot
cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(len(class_names))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('onlyMSB')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix plot
plt.savefig('onlyMSB.png')
plt.show()

