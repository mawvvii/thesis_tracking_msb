import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip
import json  # For saving sensitivity scores
import logging

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
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
log_and_print("Model and tokenizer loaded successfully.")

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
log_and_print(f"Class names: {class_names}")

# Tokenize the class names (text inputs for the CLIP model)
log_and_print("Tokenizing class names...")
text_inputs = [tokenizer.encode(class_name) for class_name in class_names]
text_inputs = torch.tensor(text_inputs).to('cuda')
log_and_print(f"Text inputs tokenized and moved to CUDA. Shape: {text_inputs.shape}")

# Move model to CUDA
log_and_print("Moving model to CUDA...")
model = model.to('cuda')
model.eval()
log_and_print("Model moved to CUDA and set to evaluation mode.")

# Clear GPU cache
log_and_print("Clearing GPU cache...")
torch.cuda.empty_cache()

# Update the transformation to include ToTensor()
log_and_print("Setting up image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert PIL images to tensors
])
log_and_print("Image transformations set.")

# Load CIFAR-10 test data with updated transformation
log_and_print("Loading CIFAR-10 test dataset...")
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
log_and_print(f"Test dataset loaded. Number of samples: {len(test_dataset)}")

# Create a DataLoader for the test dataset
log_and_print("Creating DataLoader for the test dataset...")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
log_and_print("DataLoader created.")

def load_sensitivity_scores(filepath):
    """Function to load previously saved sensitivity scores from a JSON file."""
    log_and_print(f"Loading sensitivity scores from {filepath}...")
    with open(filepath, 'r') as f:
        sensitivity_scores = json.load(f)
    log_and_print("Sensitivity scores loaded.")
    return sensitivity_scores

def convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold):
    log_and_print(f"Converting layers with sensitivity scores below {threshold} to float16...")
    for name, module in model.named_modules():
        # Get the sensitivity score for this module
        score = sensitivity_scores.get(name, None)
        if score is not None and score < threshold:
            # Skip layer normalization layers in the text encoder
            if 'ln' in name and 'text' in name:
                log_and_print(f"Skipping {name} to keep in float32")
                continue
            # Convert eligible layers to float16
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.float32:
                    log_and_print(f"Converting {name}.{param_name} to float16")
                    param.data = param.data.half()
    log_and_print("Layer conversion complete.")

def revert_text_encoder_layer_norms_to_float32(model):
    log_and_print("Reverting text encoder layer norms to float32...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and 'text' in name:
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.float16:
                    log_and_print(f"Reverting {name}.{param_name} to float32")
                    param.data = param.data.float()
    log_and_print("Reversion of layer norms complete.")

def verify_text_encoder_layer_norms_dtype(model):
    log_and_print("Verifying data types of text encoder layer norms...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and 'text' in name:
            for param_name, param in module.named_parameters(recurse=False):
                log_and_print(f"{name}.{param_name} dtype: {param.dtype}")

def list_model_layers(model, description):
    """Function to list all layers and their data types."""
    log_and_print(f"Listing model layers {description} conversion:")
    total_layers = 0
    float16_layers = 0
    for name, param in model.named_parameters():
        total_layers += 1
        dtype = param.dtype
        if dtype == torch.float16:
            float16_layers += 1
        log_and_print(f"Layer: {name}, Dtype: {dtype}")
    log_and_print(f"Total layers: {total_layers}")
    log_and_print(f"Layers in float16: {float16_layers}")
    log_and_print(f"Layers in float32: {total_layers - float16_layers}")

def display_sample_weights(model, num_layers=5):
    """Function to display sample weights from the model."""
    log_and_print(f"Displaying sample weights from {num_layers} layers:")
    count = 0
    for name, param in model.named_parameters():
        if count >= num_layers:
            break
        log_and_print(f"Layer: {name}, Weights sample: {param.data.view(-1)[:5]}")
        count += 1

# Load sensitivity scores
sensitivity_scores = load_sensitivity_scores("sensitivity_scores.json")

# List model layers before conversion
list_model_layers(model, "before")

# Optionally display sample weights before conversion
# display_sample_weights(model, num_layers=5)

# Convert layers with sensitivity scores below the threshold to float16
convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold=10)

# Revert text encoder layer norms to float32
revert_text_encoder_layer_norms_to_float32(model)

# Verify data types of layer norms
verify_text_encoder_layer_norms_dtype(model)

# List model layers after conversion
list_model_layers(model, "after")

# Optionally display sample weights after conversion
# display_sample_weights(model, num_layers=5)

def evaluate_model_accuracy_mixed_precision(model, dataloader, text_inputs):
    log_and_print("Starting model evaluation with mixed precision...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Encode text without autocast to ensure correct precision
        with torch.amp.autocast(device_type='cuda', enabled=False):
            log_and_print("Encoding text inputs...")
            text_features = model.encode_text(text_inputs)
            log_and_print(f"Text features encoded. Shape: {text_features.shape}, Dtype: {text_features.dtype}")
        
        # Proceed with evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            log_and_print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            log_and_print(f"Inputs moved to CUDA. Shape: {inputs.shape}, Dtype: {inputs.dtype}")
            log_and_print(f"Labels moved to CUDA. Shape: {labels.shape}, Dtype: {labels.dtype}")

            with autocast(device_type='cuda'):
                log_and_print("Encoding image inputs with mixed precision...")
                image_features = model.encode_image(inputs)
                log_and_print(f"Image features encoded. Shape: {image_features.shape}, Dtype: {image_features.dtype}")

            # Normalize features and compute similarity
            image_features = image_features.float()
            text_features = text_features.float()
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            log_and_print(f"Similarity computed. Shape: {similarity.shape}")

            predicted = similarity.argmax(dim=1)
            total += labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            log_and_print(f"Batch {batch_idx + 1} accuracy: {batch_correct / labels.size(0) * 100:.2f}%")

    accuracy = 100 * correct / total
    log_and_print(f"Final accuracy: {accuracy:.2f}%")
    return accuracy

# Proceed with evaluation
new_accuracy = evaluate_model_accuracy_mixed_precision(model, test_loader, text_inputs)
log_and_print(f"New Accuracy after converting some layers to float16: {new_accuracy:.2f}%")
