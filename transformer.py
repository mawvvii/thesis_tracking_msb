import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import open_clip
from torchsummary import summary
import json  # For saving sensitivity scores
import logging

# Setup logging to a file
logging.basicConfig(filename='sensitivity_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message):
    """Helper function to log and print messages."""
    print(message)
    logging.info(message)

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
text_inputs = [tokenizer.encode(class_name) for class_name in class_names]

# Convert tokenized inputs into a PyTorch tensor and move to 'cuda'
text_inputs = torch.tensor(text_inputs).to('cuda')

# Debugging: Check dtype of tokenized text inputs
print(f"Tokenized text inputs dtype: {text_inputs.dtype}")

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

# Update transformation to resize but don't convert to tensor yet (preprocessing will handle it)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
])

# Load CIFAR-10 test data as PIL images
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define a custom collate function to apply CLIP preprocessing (preprocess_val)
def custom_collate_fn(batch):
    # Unpack batch (list of tuples (image, label))
    images, labels = zip(*batch)
    
    # Apply CLIP preprocessing to each image
    images = [preprocess_val(image) for image in images]
    
    # Stack tensors to create a batch
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

# Load CIFAR-10 test data with the custom collate function
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False, collate_fn=custom_collate_fn)

from copy import deepcopy
import torch
from torch.cuda.amp import autocast

# Updated function with AMP for efficient evaluation
def evaluate_clip_model_accuracy_mixed_precision(model, dataloader, text_inputs):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Encode text inputs only once outside the loop in full precision
        text_features = model.encode_text(text_inputs).float()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # Use autocast to leverage mixed precision during image encoding
            with autocast():
                image_features = model.encode_image(inputs)

            # Convert image features back to float32 for comparison
            image_features = image_features.float()

            # Compare image features to text features to get similarity scores
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            predicted = similarity.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def parameter_sensitivity_analysis_amp(model, dataloader, text_inputs, perturbation_std=0.1):
    """
    Perform sensitivity analysis by perturbing the model parameters 
    and measuring the impact on the model's performance using AMP for mixed precision.
    
    Args:
    - model: CLIP model.
    - dataloader: DataLoader for evaluation.
    - text_inputs: Encoded text input for the CLIP model.
    - perturbation_std: Standard deviation for Gaussian noise added to weights.
    
    Returns:
    - sensitivity_scores: A dictionary containing the sensitivity scores.
    """
    # Deep copy of the original model to preserve the unmodified version
    original_model = deepcopy(model)
    original_model.eval()

    # Evaluate the original model accuracy with AMP enabled
    baseline_accuracy = evaluate_clip_model_accuracy_mixed_precision(original_model, dataloader, text_inputs)
    log_and_print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    sensitivity_scores = {}
    
    # Iterate over the model's parameters and perturb each one
    for name, param in model.named_parameters():
        if 'weight' in name:  # Focus on weights only for sensitivity analysis
            log_and_print(f"Analyzing sensitivity for layer: {name}")
            
            # Deep copy of the model for perturbation
            perturbed_model = deepcopy(original_model)

            # Add Gaussian noise to perturb the weights
            noise = torch.randn_like(param) * perturbation_std
            perturbed_weights = param.data + noise

            # Log the initial parameter values and the noise applied (for debugging)
            log_and_print(f"Initial {name} values (first 5): {param.data.view(-1)[:5]}")
            log_and_print(f"Noise applied to {name} (first 5): {noise.view(-1)[:5]}")
            log_and_print(f"Perturbed {name} values (first 5): {perturbed_weights.view(-1)[:5]}")

            # Apply the perturbed weights to the model
            perturbed_model.state_dict()[name].copy_(perturbed_weights)

            # Evaluate the perturbed model's accuracy using AMP
            perturbed_accuracy = evaluate_clip_model_accuracy_mixed_precision(perturbed_model, dataloader, text_inputs)
            log_and_print(f"Accuracy after perturbing {name}: {perturbed_accuracy:.2f}%")

            # Calculate the sensitivity score as the absolute difference from the baseline accuracy
            sensitivity_score = abs(baseline_accuracy - perturbed_accuracy)
            sensitivity_scores[name] = sensitivity_score
            log_and_print(f"Sensitivity Score for {name}: {sensitivity_score:.2f}")

    # Sort sensitivity scores in descending order
    sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    log_and_print("Sorted Sensitivity Scores (Descending):")
    for name, score in sorted_scores:
        log_and_print(f"{name}: {score:.2f}")
    # Save the sensitivity scores to a JSON file
    save_path="sensitivity_scores.json"
    with open(save_path, 'w') as f:
        json.dump(sensitivity_scores, f)
    log_and_print(f"Sensitivity scores saved to {save_path}")

    return sensitivity_scores

def load_sensitivity_scores(filepath):
    """Function to load previously saved sensitivity scores from a JSON file."""
    with open(filepath, 'r') as f:
        sensitivity_scores = json.load(f)
    return sensitivity_scores

# Example usage for sensitivity analysis
# sensitivity_scores = parameter_sensitivity_analysis_amp(model, test_loader, text_inputs, perturbation_std=0.1)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPrecisionAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, *args, **kwargs):
        # Ensure inputs are converted to float16 if weights are in float16
        if self.in_proj_weight.dtype == torch.float16:
            query = query.half()
            key = key.half()
            value = value.half()

        return super(MixedPrecisionAttention, self).forward(query, key, value, *args, **kwargs)

# Custom LayerNorm to ensure input and parameters are in float16
class MixedPrecisionLayerNorm(nn.LayerNorm):
    def forward(self, input):
        # Ensure the input is in float16 if necessary
        if input.dtype == torch.float32:
            input = input.half()

        # Ensure that the weight and bias are in float16 as well
        if self.weight.dtype == torch.float32:
            self.weight = torch.nn.Parameter(self.weight.half())  # Convert weights to float16
        if self.bias.dtype == torch.float32:
            self.bias = torch.nn.Parameter(self.bias.half())  # Convert bias to float16

        # Ensure parameters are on the same device as input
        if self.weight.device != input.device:
            self.weight = torch.nn.Parameter(self.weight.to(input.device))
            self.bias = torch.nn.Parameter(self.bias.to(input.device))

        return super(MixedPrecisionLayerNorm, self).forward(input)


def convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold=10):
    with torch.no_grad():  # Disable gradients during conversion
        for layer_name, sensitivity in sensitivity_scores.items():
            if sensitivity < threshold:
                print(f"Converting all parameters in {layer_name} to float16 due to low sensitivity score of {sensitivity:.2f}")
                
                # Split layer_name to navigate through the module hierarchy
                layer_name_parts = layer_name.split('.')
                sub_module = model
                for part in layer_name_parts[:-1]:  # Traverse the layers
                    sub_module = getattr(sub_module, part)

                # Convert all relevant parameters to float16
                # Loop through all parameters of this sub_module
                for param_name, param in sub_module.named_parameters(recurse=False):
                    print(f"Converting {layer_name}.{param_name} to float16")
                    param.data = param.data.half()

                # If sub_module has buffers (like running_mean in BatchNorm), convert them as well
                for buffer_name, buffer in sub_module.named_buffers(recurse=False):
                    print(f"Converting {layer_name}.{buffer_name} to float16")
                    buffer.data = buffer.data.half()


import torch
from torch.amp import autocast

def evaluate_model_accuracy_mixed_precision(model, dataloader, text_inputs):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Ensure text inputs are in full precision after encoding
        text_features = model.encode_text(text_inputs)
        
        # Debugging: Check dtype of text features after encoding
        print(f"Text features dtype after encoding: {text_features.dtype}")
        
        # Ensure the text features are in float32 for compatibility with mixed precision
        if text_features.dtype == torch.float16:
            text_features = text_features.float()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # Use autocast for mixed precision during image encoding and other operations
            with autocast():
                # Image encoding in mixed precision
                image_features = model.encode_image(inputs)

                # Convert image features to float32 for compatibility with text features
                image_features = image_features.float()

                # Compute similarity in float32
                similarity = (image_features @ text_features.T).softmax(dim=-1)

            predicted = similarity.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Example usage:
# After saving the sensitivity scores, you can later load and use them:
sensitivity_scores = load_sensitivity_scores("sensitivity_scores.json")
# Convert layers with sensitivity scores below the threshold to float16
# Example usage
convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold=10)
