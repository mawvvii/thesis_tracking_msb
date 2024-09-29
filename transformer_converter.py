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

def evaluate_mixed_precision_model(model, data_loader, tokenizer, labels):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    
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
    
    # Final accuracy
    accuracy = correct_predictions / total_samples
    log_and_print(f"Mixed precision accuracy: {accuracy * 100:.2f}%")
    return accuracy


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



# # Tokenize the prompts
# text_inputs = tokenizer(prompts).to('cuda')
# log_and_print(f"Text inputs tokenized and moved to CUDA. Shape: {text_inputs.shape}")

# Clear GPU cache
log_and_print("Clearing GPU cache...")
torch.cuda.empty_cache()

# # Update the transformation to include normalization
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#                          std=(0.26862954, 0.26130258, 0.27577711)),
# ])
# log_and_print("Image transformations set.")

# # Load CIFAR-10 test data with updated transformation
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# log_and_print(f"Test dataset loaded. Number of samples: {len(test_dataset)}")

# # Create a DataLoader for the test dataset
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# log_and_print("DataLoader created.")

# Inspect the model's parameters to check for biases
for name, param in model.named_parameters():
    if 'bias' in name:
        print(f"Bias found: {name}, Shape: {param.shape}")
    else:
        print(f"Weight found: {name}, Shape: {param.shape}")

# def evaluate_model_accuracy_mixed_precision(model, dataloader, text_inputs):
#     log_and_print("Starting model evaluation with mixed precision...")
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         # Check that text_inputs is in int64 as required for token embedding
#         log_and_print(f"text_inputs dtype before any changes: {text_inputs.dtype}")
#         if text_inputs.dtype != torch.int64:
#             log_and_print(f"Converting text_inputs to int64...")
#             text_inputs = text_inputs.long()
        
#         # Move text_inputs to CUDA, but ensure it remains in int64
#         text_inputs = text_inputs.to('cuda')
#         log_and_print(f"text_inputs moved to CUDA. dtype: {text_inputs.dtype}")

#         # Disable autocast here to prevent any issues with int64 inputs during encoding
#         with torch.amp.autocast(device_type='cuda', enabled=False):
#             log_and_print("Calling model.encode_text()...")
#             text_features = model.encode_text(text_inputs)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             log_and_print(f"Text features encoded. Shape: {text_features.shape}, Dtype: {text_features.dtype}")

#         # Proceed with evaluation for image inputs
#         for batch_idx, (inputs, labels) in enumerate(dataloader):
#             log_and_print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")

#             # Move inputs and labels to CUDA
#             inputs, labels = inputs.to('cuda'), labels.to('cuda')

#             with torch.amp.autocast(device_type='cuda'):
#                 log_and_print("Encoding image inputs with mixed precision...")
#                 image_features = model.encode_image(inputs)
#                 image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#                 log_and_print(f"Image features encoded. Shape: {image_features.shape}, Dtype: {image_features.dtype}")

#             # Compute similarity
#             similarity = (image_features @ text_features.T)
#             probabilities = similarity.softmax(dim=-1)
#             log_and_print(f"Similarity computed. Shape: {similarity.shape}")

#             # Get the predicted class and calculate batch accuracy
#             predicted = probabilities.argmax(dim=1)
#             total += labels.size(0)
#             batch_correct = (predicted == labels).sum().item()
#             correct += batch_correct
#             batch_accuracy = 100 * batch_correct / labels.size(0)
#             log_and_print(f"Batch {batch_idx + 1} accuracy: {batch_accuracy:.2f}%")

#     accuracy = 100 * correct / total
#     log_and_print(f"Final accuracy: {accuracy:.2f}%")
#     return accuracy



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

# # Revert text encoder layer norms to float32
# revert_text_encoder_layer_norms_to_float32(model)

# List model parameters after conversion
list_model_parameters(model, "after")

# # Verify data types of layer norms
# verify_text_encoder_layer_norms_dtype(model)

# import torch
# from torch.cuda.amp import autocast

# def evaluate_model_accuracy_amp(model, text_inputs, test_loader):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     with torch.no_grad():
#         for batch_idx, (inputs, labels) in enumerate(test_loader):
#             log_and_print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
#             log_and_print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")

#             # Move inputs and labels to CUDA
#             inputs, labels = inputs.to(device), labels.to(device)
#             log_and_print(f"Moved inputs and labels to {device}")

#             # Perform inference with autocasting
#             with torch.amp.autocast(device_type='cuda'):
#                 outputs = model(inputs, text_inputs)
#                 log_and_print(f"Model outputs: {outputs}")  # Log all model outputs

#                 # Check the number of outputs from the model
#                 if isinstance(outputs, tuple):
#                     log_and_print(f"Model returned {len(outputs)} outputs.")
#                 else:
#                     log_and_print("Model returned a single output.")

#             # Adjust here based on the actual number of outputs from the model
#             logits_per_image = outputs[0]  # Assuming the first output is the image logits
#             log_and_print(f"Logits per image shape: {logits_per_image.shape}")

#             # Apply softmax to the logits and find the predicted class
#             outputs = F.softmax(logits_per_image, dim=1)
#             log_and_print(f"Softmax outputs: {outputs[:5]}")  # Log first 5 outputs for inspection

#             # Get the predicted class by finding the index of the maximum value
#             _, predicted = torch.max(outputs, dim=1)
#             log_and_print(f"Predicted classes: {predicted[:5]}")

#             # Count the number of correct predictions
#             correct_predictions = (predicted == labels).sum().item()
#             correct += correct_predictions
#             total += labels.size(0)

#             log_and_print(f"Batch {batch_idx + 1}: {correct_predictions}/{labels.size(0)} correct predictions")
        
#     accuracy = correct / total if total > 0 else 0
#     log_and_print(f"Total correct: {correct}/{total}, Accuracy: {accuracy * 100:.2f}%")

#     return accuracy

log_and_print("Model and tokenizer loaded successfully.")
log_and_print("Starting evaluation with mixed precision...")
evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
log_and_print("Evaluation completed.")

