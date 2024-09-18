from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader, get_modified_dataloader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from torchsummary import summary
import numpy as np
from copy import deepcopy
from torch.cuda.amp import autocast
import torchvision.models as models
from torch.amp.autocast_mode import autocast  # Corrected import for AMP autocast

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the modified data
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = models.resnet34(pretrained=True)

# Move model to GPU before loading the state_dict
model.to('cuda')
model.eval()  # Set the model to evaluation mode

# Provide the correct input size
summary(model, input_size=(3, 32, 32))

# Updated function with AMP context
def evaluate_model_accuracy_amp(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            # Use autocast to handle mixed precision automatically
            with autocast():
                outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def parameter_sensitivity_analysis(model, dataloader, perturbation_std=0.1):
    """
    Perform sensitivity analysis by perturbing the model parameters 
    and measuring the impact on the model's performance.
    
    Args:
    - model: PyTorch model.
    - dataloader: DataLoader for evaluation.
    - perturbation_std: Standard deviation for Gaussian noise added to weights.
    
    Returns:
    - sensitivity_scores: A dictionary containing the sensitivity scores.
    """
    # Copy the model to keep the original model weights unchanged
    original_model = deepcopy(model)
    original_model.eval()
    
    # Evaluate the original model accuracy
    baseline_accuracy = evaluate_model_accuracy_amp(original_model, dataloader)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    
    sensitivity_scores = {}
    
    # Iterate over each layer in the model
    for name, param in model.named_parameters():
        if 'weight' in name:  # Focus on weights for sensitivity analysis
            print(f"Analyzing sensitivity for layer: {name}")
            
            # Perturb weights with Gaussian noise
            perturbed_model = deepcopy(original_model)
            
            # Add noise to the weights
            noise = torch.randn_like(param) * perturbation_std
            perturbed_weights = param.data + noise
            
            # Update model with perturbed weights
            perturbed_model.state_dict()[name].copy_(perturbed_weights)
            
            # Evaluate the perturbed model accuracy
            perturbed_accuracy = evaluate_model_accuracy_amp(perturbed_model, dataloader)
            
            # Compute the sensitivity score
            sensitivity_score = abs(baseline_accuracy - perturbed_accuracy)
            sensitivity_scores[name] = sensitivity_score
            print(f"Sensitivity Score for {name}: {sensitivity_score:.2f}")
    
    # Sort sensitivity scores in descending order
    sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    print("Sorted Sensitivity Scores (Descending):")
    for name, score in sorted_scores:
        print(f"{name}: {score:.2f}")
    
    return sensitivity_scores

# Increase perturbation_std to make changes more noticeable
sensitivity_scores = parameter_sensitivity_analysis(model, test_loader, perturbation_std=0.1)

# Function to convert selected layer weights to float16
import torch

# Function to convert all parameters of selected layers to float16
def convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold):
    with torch.no_grad():  # No need to compute gradients for this operation
        for layer_name, sensitivity in sensitivity_scores.items():
            if sensitivity < threshold:  # Check if sensitivity is below threshold
                print(f"Converting {layer_name} to float16 due to low sensitivity score of {sensitivity:.2f}")
                
                # Convert the weight and bias of this layer to float16
                layer = dict(model.named_parameters())[layer_name]
                layer.data = layer.data.half()  # Convert weights to float16
                
                # Convert other associated parameters if they exist
                # e.g., BatchNorm layers have additional parameters like bias, running_mean, and running_var
                layer_name_parts = layer_name.split('.')
                if len(layer_name_parts) > 1:
                    # Navigate to the module using the name parts
                    sub_module = model
                    for part in layer_name_parts[:-1]:  # Skip the last part to reach the module
                        sub_module = getattr(sub_module, part)
                    
                    if isinstance(sub_module, torch.nn.BatchNorm2d):
                        sub_module.bias.data = sub_module.bias.data.half()
                        sub_module.running_mean.data = sub_module.running_mean.data.half()
                        sub_module.running_var.data = sub_module.running_var.data.half()

# Example usage:
# Convert layers with sensitivity scores below the threshold to float16
threshold = 10  # Define threshold for sensitivity
convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold)

# Re-evaluate the model performance using the updated function that handles mixed precision
new_accuracy = evaluate_model_accuracy_amp(model, test_loader)
print(f"New Accuracy after converting some layers to float16: {new_accuracy:.2f}%")

save_directory = '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/SmallerModels'
onlyMsbModel = f'{save_directory}/onlyMSBmodel.pth'

# Save the model's state dict and metadata
metadata = {'float16_layers': [name for name, param in model.named_parameters() if param.dtype == torch.float16]}
torch.save({'state_dict': model.state_dict(), 'metadata': metadata}, '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/SmallerModels/onlyMSBmodel.pth')

import torch
from torch.cuda.amp import autocast, GradScaler

def compute_memory_usage_with_amp(model, input_size=(3, 32, 32)):
    # Check if CUDA is available and move model to GPU if so
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_memory = 0
    activation_memory = 0

    # Calculate memory for model parameters
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            param_memory += param.numel() * 2  # float16 is 2 bytes
        else:  # default to float32
            param_memory += param.numel() * 4  # float32 is 4 bytes

    # Dummy input for forward pass to estimate activation memory
    input_tensor = torch.randn(1, *input_size).to(device).float()  # Input should start in float32

    # Function to register hook to compute activation memory
    def hook(module, input, output):
        nonlocal activation_memory
        if isinstance(output, tuple):  # Some outputs are tuples
            for out in output:
                if out.dtype == torch.float16:
                    activation_memory += out.numel() * 2
                else:
                    activation_memory += out.numel() * 4
        else:
            if output.dtype == torch.float16:
                activation_memory += output.numel() * 2
            else:
                activation_memory += output.numel() * 4

    # Register hooks to capture memory usage
    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook))

    # Use autocast for mixed precision during the forward pass
    with autocast():  # AMP context manager
        model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Total memory consumption (in MB)
    total_memory = (param_memory + activation_memory) / (1024 ** 2)
    
    return total_memory, param_memory / (1024 ** 2), activation_memory / (1024 ** 2)

# Example usage
total_memory, param_mem, act_mem = compute_memory_usage_with_amp(model)
print(f"Total Memory: {total_memory:.2f} MB (Parameters: {param_mem:.2f} MB, Activations: {act_mem:.2f} MB)")

def custom_mixed_precision_summary(model, input_size=(3, 32, 32), device='cuda'):
    # Move model to the specified device
    model.to(device)

    print("Generating mixed precision model summary:\n")

    total_params = 0
    float32_params = 0
    float16_params = 0
    layer_details = []

    # Loop through all modules in the model
    for name, layer in model.named_modules():
        # Get the number of parameters and the data type for each layer
        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        if num_params > 0:  # Only consider layers with parameters
            total_params += num_params
            
            # Determine the data type
            param_dtype = next(layer.parameters()).dtype
            layer_info = {
                'Layer Name': name,
                'Type': type(layer).__name__,
                'Output Shape': 'N/A',  # Could compute if needed with hooks
                'Num Params': num_params,
                'Data Type': str(param_dtype)
            }
            
            # Track memory by dtype
            if param_dtype == torch.float32:
                float32_params += num_params
            elif param_dtype == torch.float16:
                float16_params += num_params

            layer_details.append(layer_info)

    # Print the layer-wise details
    print(f"{'Layer Name':<30} {'Type':<20} {'Num Params':<15} {'Data Type':<10}")
    print("-" * 80)
    for layer_info in layer_details:
        print(f"{layer_info['Layer Name']:<30} {layer_info['Type']:<20} {layer_info['Num Params']:<15} {layer_info['Data Type']:<10}")
    
    # Print the summary of parameters
    print("\nModel Summary with Mixed Precision Details:")
    print(f"Total Parameters: {total_params}")
    print(f"Float32 Parameters: {float32_params} ({float32_params * 4 / (1024**2):.2f} MB)")
    print(f"Float16 Parameters: {float16_params} ({float16_params * 2 / (1024**2):.2f} MB)")

# Example Usage
custom_mixed_precision_summary(model)

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

# Proceed with evaluation
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
df.to_excel('resnet34_onlyMSB.xlsx', index=False)

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
plt.savefig('Confusion_Matrix_resenet34_onlyMSB.png')
plt.show()

