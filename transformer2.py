import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import open_clip
from torchsummary import summary
from torch.amp import autocast

# Load the CLIP model and its corresponding tokenizer from Hugging Face
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Tokenize the class names (text inputs for the CLIP model)
text_inputs = tokenizer(class_names)

# Move model and tokenized text inputs to CUDA
model = model.to('cuda')
text_inputs = text_inputs.to('cuda')

model.eval()

# Move the model to the GPU (or CPU if not available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# # Print the model summary with input size (3 channels, 224x224 image size)
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Data Type: {param.dtype}")

# Clear GPU cache
torch.cuda.empty_cache()

from torchvision import transforms

# Update the transformation to include ToTensor()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert PIL images to tensors
])

# Load CIFAR-10 test data with updated transformation
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


import torch
import torch.nn as nn

import torch
from copy import deepcopy
from torch.cuda.amp import autocast

# Function to evaluate model accuracy using autocast
def evaluate_model_accuracy_amp(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            # Use autocast for mixed precision
            with autocast():
                outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Sensitivity Analysis focused on weights only
def parameter_sensitivity_analysis_weights_only(model, dataloader, perturbation_std=0.1):
    """
    Perform sensitivity analysis by perturbing only the weights 
    and measuring the impact on model's accuracy.
    """
    # Deep copy of the original model
    original_model = deepcopy(model)
    original_model.eval()
    
    # Evaluate the original model accuracy
    baseline_accuracy = evaluate_model_accuracy_amp(original_model, dataloader)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    
    sensitivity_scores = {}
    
    # Iterate over each parameter in the model
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only perturb the weights
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

# Example usage:
# Load your model, dataloader, and then run the sensitivity analysis
sensitivity_scores = parameter_sensitivity_analysis_weights_only(model, test_dataset, perturbation_std=0.1)

