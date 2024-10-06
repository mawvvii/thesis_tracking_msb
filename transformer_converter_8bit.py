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
import pandas as pd
import torch
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def save_metrics_to_file(result_metrics, file_path):
    with open(file_path, 'w') as f:
        # Write the metrics to the file
        f.write(f"Accuracy: {result_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Accuracy Metrics: {result_metrics['accuracy_metric'] * 100:.2f}%\n")
        f.write(f"Precision: {result_metrics['precision'] * 100:.2f}%\n")
        f.write(f"Recall: {result_metrics['recall'] * 100:.2f}%\n")
        f.write(f"F1-score: {result_metrics['f1_score'] * 100:.2f}%\n")
        f.write(f"Mean IoU (mIoU): {result_metrics['miou']:.4f}\n")
        f.write(f"Memory allocated: {result_metrics['memory_allocated']:.2f} MB\n")
        f.write(f"Memory reserved: {result_metrics['memory_reserved']:.2f} MB\n")
        f.write(f"Initial_memory_allocated: {result_metrics['Initial_memory_allocated']:.2f} MB\n")
        f.write(f"Initial_memory_reserved: {result_metrics['Initial_memory_reserved']:.2f} MB\n")
        f.write(f"Final_memory_allocated: {result_metrics['Final_memory_allocated']:.2f} MB\n")
        f.write(f"Final_memory_reserved: {result_metrics['Final_memory_reserved']:.2f} MB\n")
        
        # Write IoU for each class
        f.write("IoU per class:\n")
        for idx, iou in enumerate(result_metrics['iou_list']):
            f.write(f"Class '{class_names[idx]}': IoU = {iou:.4f}\n")
        

def evaluate(model, data_loader, tokenizer, labels, class_names):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    top3_probs_list = []
    top3_classes_list = []

    # Track initial memory usage
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
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
            labels_batch = labels_batch.to('cuda')

            # Autocast for mixed precision handling
            with autocast('cuda', dtype=torch.float16):
                image_features = model.encode_image(images.to('cuda'))  # Image embeddings
                image_features = F.normalize(image_features, dim=-1)  # Normalize to unit vectors
                
            # Compute cosine similarity between image and text features
            similarities = torch.matmul(image_features, text_features.T)  # (Batch_size, num_text_labels)
            
            # Get top 3 predictions
            top3_probs, top3_classes = torch.topk(similarities, 3, dim=1)
            top3_probs_list.extend(top3_probs.cpu().numpy())  # Save top 3 probabilities
            top3_classes_list.extend(top3_classes.cpu().numpy())  # Save top 3 classes
            
            # Get predictions by finding the text label with the highest similarity score for each image
            predictions = torch.argmax(similarities, dim=1)
            
            # Store predictions and actual labels for metric calculation later
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

            # Calculate batch accuracy
            correct_predictions += torch.sum(predictions == labels_batch).item()
            total_samples += labels_batch.size(0)

    # Track memory usage after inference
    final_memory_allocated = torch.cuda.memory_allocated()
    final_memory_reserved = torch.cuda.memory_reserved()

    # Get peak memory usage during inference
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Peak memory in MB
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # Peak memory in MB

    # Calculate and log the memory usage difference
    memory_allocated = (final_memory_allocated - initial_memory_allocated) / (1024 ** 2)  # Convert to MB
    memory_reserved = (final_memory_reserved - initial_memory_reserved) / (1024 ** 2)  # Convert to MB
    log_and_print(f"Memory allocated during inference: {memory_allocated:.2f} MB")
    log_and_print(f"Memory reserved during inference: {memory_reserved:.2f} MB")
    log_and_print(f"Peak memory allocated during inference: {peak_memory_allocated:.2f} MB")
    log_and_print(f"Peak memory reserved during inference: {peak_memory_reserved:.2f} MB")
    
    # Final accuracy
    accuracy = correct_predictions / total_samples
    # Calculate other metrics: precision, recall, F1-score
    accuracy_metric = accuracy_score(all_labels, all_preds)
    precision_metric = precision_score(all_labels, all_preds, average='weighted')
    recall_metric = recall_score(all_labels, all_preds, average='weighted')
    f1_metric = f1_score(all_labels, all_preds, average='weighted')
    log_and_print(f"Mixed precision accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    # Calculate IoU (Intersection over Union) and mIoU (Mean IoU)
    iou_list = []
    for i in range(len(class_names)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        union = TP + FP + FN
        iou = TP / union if union > 0 else 0
        iou_list.append(iou)
        log_and_print(f"Class '{class_names[i]}': IoU = {iou:.4f}")

    miou = np.mean(iou_list)
    log_and_print(f"Mean IoU (mIoU): {miou:.4f}")
    
    # Return metrics and top-3 predictions
    return {
        "accuracy": accuracy,
        "accuracy_metric": accuracy_metric,
        "precision": precision_metric,
        "recall": recall_metric,
        "f1_score": f1_metric,
        "miou": miou,
        "iou_list": iou_list,
        "top3_probs": top3_probs_list,
        "top3_classes": top3_classes_list,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "memory_allocated": memory_allocated,
        "memory_reserved": memory_reserved,
        "peak_memory_allocated": peak_memory_allocated,
        "peak_memory_reserved": peak_memory_reserved,
        "Initial_memory_allocated":initial_memory_allocated,
        "Initial_memory_reserved":initial_memory_reserved,
        "Final_memory_allocated":final_memory_allocated,
        "Final_memory_reserved":final_memory_reserved,
    }

def save_results_to_excel(result_metrics, class_names, file_path='results.xlsx'):
    # Prepare data for the Excel file
    data = []
    all_labels = result_metrics["all_labels"]
    all_preds = result_metrics["all_preds"]
    top3_probs = result_metrics["top3_probs"]
    top3_classes = result_metrics["top3_classes"]

    for idx, (actual, predicted, probs, classes) in enumerate(zip(all_labels, all_preds, top3_probs, top3_classes)):
        top1_prob, top2_prob, top3_prob = probs
        top1_class, top2_class, top3_class = [class_names[i] for i in classes]

        data.append([
            class_names[actual],  # Actual class
            class_names[predicted],  # Predicted class
            top1_prob,  # Top 1 probability
            top1_class,  # Top 1 class
            top2_prob,  # Top 2 probability
            top2_class,  # Top 2 class
            top3_prob,  # Top 3 probability
            top3_class,  # Top 3 class
        ])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=[
        'Actual', 'Predicted', 'Top1_Prob', 'Top1_Class', 'Top2_Prob', 'Top2_Class', 'Top3_Prob', 'Top3_Class'
    ])
    
    # Save to Excel
    df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")

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

def convert_parameters_below_threshold_to_8bit(model, sensitivity_scores, threshold):
    log_and_print(f"Converting weights and corresponding biases with sensitivity scores below {threshold} to 8-bit precision, skipping layers with 'ln'...")
    converted_params = 0

    for name, param in model.named_parameters():
        score = sensitivity_scores.get(name, None)
        if score is not None and score < threshold:
            if 'weight' in name:
                log_and_print(f"Converting weight: '{name}' to 8-bit precision")
                # Extract the most significant 8 bits (MSB) from each 32-bit float
                if param.dtype == torch.float32:
                    param_data = param.data
                    # Convert float32 to int32, shift right by 24 bits to keep the most significant 8 bits
                    msb = (param_data.view(torch.int32) >> 24).to(torch.float32)
                    param.data = msb  # Replace the original parameter with the 8-bit version
                    log_and_print(f"Converted '{name}' to 8-bit precision")
                    converted_params += 1

                # Now automatically convert the corresponding bias for this layer if it exists
                bias_name = name.replace('weight', 'bias')
                if bias_name in dict(model.named_parameters()):
                    bias_param = dict(model.named_parameters())[bias_name]
                    log_and_print(f"Found corresponding bias: '{bias_name}'")
                    if bias_param.dtype == torch.float32:
                        bias_data = bias_param.data
                        msb_bias = (bias_data.view(torch.int32) >> 24).to(torch.float32)
                        bias_param.data = msb_bias  # Replace bias with its 8-bit version
                        log_and_print(f"Converted bias '{bias_name}' to 8-bit precision")
                        converted_params += 1

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

# log_and_print("Model and tokenizer loaded successfully.")
log_and_print("Starting evaluation before float 16 ")
# evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
# log_and_print("Evaluation completed.")

# Example usage after model evaluation
result_metrics = evaluate(model, data_loader, tokenizer, labels, class_names)

# Save the results to an Excel file
save_results_to_excel(result_metrics, class_names, 'before_conversion_evaluation_results.xlsx')

# # Save the results to a text file
save_metrics_to_file(result_metrics, 'before_conversion_evaluation_results.txt')

# Optional: print a confirmation message
print("before_conversion_evaluation_results.xlsx")

# # List model parameters before conversion
# list_model_parameters(model, "before")

# Convert parameters with sensitivity scores below the threshold to float16
convert_parameters_below_threshold_to_8bit(model, sensitivity_scores, threshold=0)

# List model parameters after conversion
list_model_parameters(model, "after")

# log_and_print("Model and tokenizer loaded successfully.")
# log_and_print("Starting evaluation with mixed precision...")
# evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
# log_and_print("Evaluation completed.")

# Free up memory before starting evaluation
torch.cuda.empty_cache()
gc.collect()

# Example usage after model evaluation
result_metrics = evaluate(model, data_loader, tokenizer, labels, class_names)

# Save the results to an Excel file
save_results_to_excel(result_metrics, class_names, 'evaluation_results.xlsx')

# # Save the results to a text file
save_metrics_to_file(result_metrics, 'evaluation_results.txt')

# Optional: print a confirmation message
print("evaluation_results.xlsx")