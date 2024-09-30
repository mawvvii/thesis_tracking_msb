import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip
import logging
import json 
import torch
from torch.amp import autocast
import torch.nn.functional as F

import torch
from torch.amp import autocast
import torch.nn.functional as F

def evaluate_mixed_precision_model(model, data_loader, tokenizer, labels):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    # Track initial memory usage
    initial_memory_allocated = torch.cuda.memory_allocated()
    initial_memory_reserved = torch.cuda.memory_reserved()
    
    log_and_print(f"Initial memory allocated: {initial_memory_allocated / (1024 ** 2):.2f} MB")
    log_and_print(f"Initial memory reserved: {initial_memory_reserved / (1024 ** 2):.2f} MB")
    
    with torch.no_grad():
        # Tokenize all the possible labels beforehand
        text_inputs = tokenize_labels(labels)
        text_inputs = torch.cat(text_inputs)  # Stack the text inputs
        
        # Normalize text embeddings as CLIP does for similarity computation
        with autocast('cuda', dtype=torch.float16):
            text_features = model.encode_text(text_inputs)  # Text embeddings
            text_features = F.normalize(text_features, dim=-1)  # Normalize to unit vectors

        for batch in data_loader:
            images, labels_batch = preprocess_batch(batch)  # Rename 'labels' to avoid conflict
            
            # Autocast for mixed precision handling
            with autocast('cuda', dtype=torch.float16):
                image_features = model.encode_image(images)  # Image embeddings
                image_features = F.normalize(image_features, dim=-1)  # Normalize to unit vectors
                
            # Compute cosine similarity between image and text features
            similarities = torch.matmul(image_features, text_features.T)  # (Batch_size, num_text_labels)
            
            # Get predictions by finding the text label with the highest similarity score for each image
            predictions = torch.argmax(similarities, dim=1)
            
            # Calculate accuracy
            correct_predictions += torch.sum(predictions == labels_batch).item()
            total_samples += labels_batch.size(0)

    # Track memory usage after inference
    final_memory_allocated = torch.cuda.memory_allocated()
    final_memory_reserved = torch.cuda.memory_reserved()

    # Calculate and log the memory usage difference
    memory_allocated = (final_memory_allocated - initial_memory_allocated) / (1024 ** 2)  # Convert to MB
    memory_reserved = (final_memory_reserved - initial_memory_reserved) / (1024 ** 2)  # Convert to MB
    log_and_print(f"Memory allocated during inference: {memory_allocated:.2f} MB")
    log_and_print(f"Memory reserved during inference: {memory_reserved:.2f} MB")
    
    # Final accuracy
    accuracy = correct_predictions / total_samples
    log_and_print(f"Mixed precision accuracy: {accuracy * 100:.2f}%")
    
    # Return accuracy and memory usage
    return accuracy, memory_allocated, memory_reserved

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)

def log_and_print(message):
    """Helper function to log and print messages."""
    logging.info(message)

# Load the CLIP model and its corresponding tokenizer
log_and_print("Loading the CLIP model and tokenizer...")
model, _, _ = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    device='cuda'
)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
log_and_print("Model and tokenizer loaded successfully.")

# Example for loading a dataset, e.g., CIFAR-10 for image classification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Example: Loading a dataset (adjust according to your needs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP models expect this size
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

log_and_print("Dataset loaded successfully.")

def preprocess_batch(batch):
    images, labels = batch
    return images.to('cuda', dtype=torch.float16), labels.to('cuda')


def tokenize_text(text):
    # Tokenize the text prompt
    return tokenizer([text]).to('cuda')
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def tokenize_labels(labels):
    return [tokenizer(f"A photo of a {label}.").to('cuda') for label in labels]


# Clear GPU cache
log_and_print("Clearing GPU cache...")
torch.cuda.empty_cache()

# Inspect the model's parameters to check for biases
for name, param in model.named_parameters():
    if 'bias' in name:
        print(f"Bias found: {name}, Shape: {param.shape}")
    else:
        print(f"Weight found: {name}, Shape: {param.shape}")


def load_sensitivity_scores(filepath):
    """Function to load previously saved sensitivity scores from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            sensitivity_scores = json.load(f)
        log_and_print("Sensitivity scores loaded successfully.")
        if not sensitivity_scores:
            log_and_print("Warning: Sensitivity scores dictionary is empty.")
    except Exception as e:
        log_and_print(f"Error loading sensitivity scores: {e}")
        sensitivity_scores = {}
    return sensitivity_scores

def convert_parameters_below_threshold_to_float16(model, sensitivity_scores, threshold):
    log_and_print(f"Converting weights and corresponding biases with sensitivity scores below {threshold} to float16, skipping layers with 'ln'...")
    converted_params = 0

    # Collect parameter names and normalize them
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)
    normalized_param_names = [name.replace('.', '_').lower() for name in param_names]
    normalized_to_original_param_names = {normalized_name: original_name for normalized_name, original_name in zip(normalized_param_names, param_names)}

    # Normalize sensitivity keys
    sensitivity_keys = list(sensitivity_scores.keys())
    normalized_sensitivity_keys = [key.replace('.', '_').lower() for key in sensitivity_keys]
    normalized_to_original_sensitivity_keys = {normalized_key: original_key for normalized_key, original_key in zip(normalized_sensitivity_keys, sensitivity_keys)}

    # Find matching names for conversion
    matching_names = set(normalized_param_names) & set(normalized_sensitivity_keys)
    logging.info(f"Number of matching parameter names: {len(matching_names)}")

    for normalized_name in matching_names:
        param_name = normalized_to_original_param_names[normalized_name]
        sensitivity_key = normalized_to_original_sensitivity_keys[normalized_name]
        score = sensitivity_scores.get(sensitivity_key, None)

        if score is not None:
            # # Skip layers containing 'ln'
            # if 'ln' in param_name:
            #     log_and_print(f"Skipping layer '{param_name}' due to 'ln' in the name")
            #     continue
            
            log_and_print(f"Parameter: '{param_name}', Sensitivity score: {score}")
            if score < threshold:
                # Convert weight
                if 'weight' in param_name:
                    log_and_print(f"Converting weight: '{param_name}'")
                    weight_param = dict(model.named_parameters())[param_name]
                    if weight_param.dtype == torch.float32:
                        weight_param.data = weight_param.data.half()
                        log_and_print(f"Converted '{param_name}' to float16")
                        converted_params += 1
                    
                    # Now automatically convert the corresponding bias for this layer if it exists
                    bias_name = param_name.replace('weight', 'bias')
                    if bias_name in dict(model.named_parameters()):
                        bias_param = dict(model.named_parameters())[bias_name]
                        log_and_print(f"Found corresponding bias: '{bias_name}'")
                        if bias_param.dtype == torch.float32:
                            bias_param.data = bias_param.data.half()
                            log_and_print(f"Converted bias '{bias_name}' to float16")
                            converted_params += 1
                        else:
                            log_and_print(f"Skipping bias '{bias_name}' as it is not float32")
                    else:
                        log_and_print(f"No bias found for '{param_name}'")

    log_and_print(f"Parameter conversion complete. Total parameters converted: {converted_params}")

def revert_text_encoder_layer_norms_to_float32(model):
    log_and_print("Reverting text encoder layer norms to float32...")
    for name, param in model.named_parameters():
        if 'text' in name and 'ln' in name:
            if param.dtype == torch.float16:
                log_and_print(f"Reverting '{name}' to float32")
                param.data = param.data.float()
    log_and_print("Reversion of layer norms complete.")

def verify_text_encoder_layer_norms_dtype(model):
    log_and_print("Verifying data types of text encoder layer norms...")
    for name, param in model.named_parameters():
        print(f"name{name}, Dtype: {param.dtype} ")
        if 'text' in name and 'ln' in name:
            log_and_print(f"Parameter: '{name}', Dtype: {param.dtype}")

def list_model_parameters(model, description):
    """Function to list all parameters and their data types."""
    log_and_print(f"Listing model parameters {description} conversion:")
    total_params = 0
    float16_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        dtype = param.dtype
        if dtype == torch.float16:
            float16_params += 1
        # log_and_print(f"Parameter: {name}, Dtype: {dtype}")
    log_and_print(f"Total parameters: {total_params}")
    log_and_print(f"Parameters in float16: {float16_params}")
    log_and_print(f"Parameters in float32: {total_params - float16_params}")

# Load sensitivity scores
sensitivity_scores = load_sensitivity_scores("sensitivity_scores.json")

log_and_print("Model and tokenizer loaded successfully.")
log_and_print("Starting evaluation before float 16 ")
evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
log_and_print("Evaluation completed.")

# List model parameters before conversion
list_model_parameters(model, "before")

# Convert parameters with sensitivity scores below the threshold to float16
convert_parameters_below_threshold_to_float16(model, sensitivity_scores, threshold=10)

# List model parameters after conversion
list_model_parameters(model, "after")

log_and_print("Model and tokenizer loaded successfully.")
log_and_print("Starting evaluation with mixed precision...")
evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
log_and_print("Evaluation completed.")

