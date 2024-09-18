import torch
from torch.amp import autocast
from torchsummary import summary
import open_clip
from copy import deepcopy
from utils.dataset import get_dataloader


# Load the CLIP model and its corresponding tokenizer from Hugging Face
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
)
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

# # Move model to GPU and set it to evaluation mode
# with autocast():
#     model.to('cuda')
model = model.to('cpu')
model.to('cuda', non_blocking=True)  # Load parts back to the GPU when needed

model.eval()

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Tokenize the class names (text inputs for the CLIP model)
text_inputs = tokenizer(class_names)

torch.cuda.empty_cache()

# Preprocess the CIFAR-10 test images (image inputs for the CLIP model)
test_loader = get_dataloader('CIFAR10', 'test', batch_size=50, shuffle=False)

# Check model summary
summary(model, input_size=(3, 32, 32))

# Updated evaluation function with AMP and text input comparison
def evaluate_clip_model_accuracy_amp(model, dataloader, text_inputs):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # Preprocess the inputs using CLIP's transformations
            inputs = preprocess_val(inputs)
            
            # Use autocast for mixed precision
            with autocast():
                image_features = model.encode_image(inputs)
                text_features = model.encode_text(text_inputs)

            # Compare image features to text features to get similarity scores
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            predicted = similarity.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Function for parameter sensitivity analysis
def parameter_sensitivity_analysis(model, dataloader, text_inputs, perturbation_std=0.1):
    original_model = deepcopy(model)
    original_model.eval()

    # Evaluate the original model accuracy
    baseline_accuracy = evaluate_clip_model_accuracy_amp(original_model, dataloader, text_inputs)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    sensitivity_scores = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Analyzing sensitivity for layer: {name}")
            
            perturbed_model = deepcopy(original_model)
            noise = torch.randn_like(param) * perturbation_std
            perturbed_weights = param.data + noise
            perturbed_model.state_dict()[name].copy_(perturbed_weights)

            # Evaluate perturbed model accuracy
            perturbed_accuracy = evaluate_clip_model_accuracy_amp(perturbed_model, dataloader, text_inputs)
            sensitivity_score = abs(baseline_accuracy - perturbed_accuracy)
            sensitivity_scores[name] = sensitivity_score
            print(f"Sensitivity Score for {name}: {sensitivity_score:.2f}")

    sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    print("Sorted Sensitivity Scores (Descending):")
    for name, score in sorted_scores:
        print(f"{name}: {score:.2f}")

    return sensitivity_scores

# Run the parameter sensitivity analysis
sensitivity_scores = parameter_sensitivity_analysis(model, test_loader, text_inputs, perturbation_std=0.1)
