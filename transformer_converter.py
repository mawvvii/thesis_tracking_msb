import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
import open_clip
import json  # For saving sensitivity scores
import logging

# Load the CLIP model and its corresponding tokenizer from OpenAI CLIP
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
text_inputs = torch.tensor(text_inputs).to('cuda')

# Move model to CUDA
model = model.to('cuda')
model.eval()

# Clear GPU cache
torch.cuda.empty_cache()

# Update the transformation to include ToTensor()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert PIL images to tensors
])

# Load CIFAR-10 test data with updated transformation
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

def load_sensitivity_scores(filepath):
    """Function to load previously saved sensitivity scores from a JSON file."""
    with open(filepath, 'r') as f:
        sensitivity_scores = json.load(f)
    return sensitivity_scores

# Setup logging to a file
logging.basicConfig(filename='sensitivity_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message):
    """Helper function to log and print messages."""
    print(message)
    logging.info(message)

def convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold):
    for name, module in model.named_modules():
        # Get the sensitivity score for this module
        score = sensitivity_scores.get(name, None)
        if score is not None and score < threshold:
            # Skip layer normalization layers in the text encoder
            if 'ln' in name and 'text' in name:
                print(f"Skipping {name} to keep in float32")
                continue
            # Convert eligible layers to float16
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.float32:
                    print(f"Converting {name}.{param_name} to float16")
                    param.data = param.data.half()

def revert_text_encoder_layer_norms_to_float32(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and 'text' in name:
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.float16:
                    print(f"Reverting {name}.{param_name} to float32")
                    param.data = param.data.float()

sensitivity_scores = load_sensitivity_scores("sensitivity_scores.json")

# Convert layers with sensitivity scores below the threshold to float16
convert_layers_below_threshold_to_float16(model, sensitivity_scores, threshold=10)

# Revert text encoder layer norms to float32
revert_text_encoder_layer_norms_to_float32(model)

def verify_text_encoder_layer_norms_dtype(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and 'text' in name:
            for param_name, param in module.named_parameters(recurse=False):
                print(f"{name}.{param_name} dtype: {param.dtype}")

verify_text_encoder_layer_norms_dtype(model)

def evaluate_model_accuracy_mixed_precision(model, dataloader, text_inputs):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        # Encode text without autocast to ensure correct precision
        with torch.amp.autocast(device_type='cuda', enabled=False):
            text_features = model.encode_text(text_inputs)
        
        # Proceed with evaluation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            with autocast(device_type='cuda'):
                image_features = model.encode_image(inputs)

            # Normalize features and compute similarity
            image_features = image_features.float()
            text_features = text_features.float()
            similarity = (image_features @ text_features.T).softmax(dim=-1)

            predicted = similarity.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Final accuracy: {accuracy}%")
    return accuracy

# Proceed with evaluation
new_accuracy = evaluate_model_accuracy_mixed_precision(model, test_loader, text_inputs)
log_and_print(f"New Accuracy after converting some layers to float16: {new_accuracy:.2f}%")
