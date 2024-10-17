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
import csv
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

# # Inspect the model's parameters to check for biases
# for name, param in model.named_parameters():
#     if 'bias' in name:
#         print(f"Bias found: {name}, Shape: {param.shape}")
#     else:
#         print(f"Weight found: {name}, Shape: {param.shape}")

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
    log_and_print(f"Converting weights and corresponding biases with sensitivity scores below {threshold} to float8, skipping layers with 'ln'...")
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
            log_and_print(f"Parameter: '{param_name}', Sensitivity score: {score}")
            if score < threshold:
                # Convert weight
                # log_and_print(f"converting .. {param_name}")
                if 'weight' in param_name:
                    log_and_print(f"Converting weight: '{param_name}'")
                    weight_param = dict(model.named_parameters())[param_name]
                    if weight_param.dtype == torch.float16:
                        weight_param.data = weight_param.data.to(torch.float8_e5m2)
                        log_and_print(f"Converted '{param_name}' to {weight_param.dtype}")
                        converted_params += 1
                    
                    # Now automatically convert the corresponding bias for this layer if it exists
                    bias_name = param_name.replace('weight', 'bias')
                    if bias_name in dict(model.named_parameters()):
                        bias_param = dict(model.named_parameters())[bias_name]
                        log_and_print(f"Found corresponding bias: '{bias_name}'")
                        if bias_param.dtype == torch.float16:
                            bias_param.data = bias_param.data.to(torch.float8_e5m2)
                            log_and_print(f"Converted bias '{bias_name}' to {bias_param.dtype}")
                            converted_params += 1
                        else:
                            log_and_print(f"Skipping bias '{bias_name}' as it is not float32")
                    else:
                        log_and_print(f"No bias found for '{param_name}'")

    log_and_print(f"Parameter conversion complete. Total parameters converted: {converted_params}")

def list_model_parameters(model, description):
    """Function to list all parameters and their data types."""
    log_and_print(f"Listing model parameters {description} conversion:")
    total_params = 0
    float16_params = 0
    float8_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        dtype = param.dtype
        if dtype == torch.float16:
            float16_params += 1
        if dtype == torch.float8_e5m2:
            float8_params += 1
        # log_and_print(f"Parameter: {name}, Dtype: {dtype}")
    log_and_print(f"Total parameters: {total_params}")
    log_and_print(f"Parameters in float16: {float16_params}")
    log_and_print(f"Parameters in float8: {float8_params}")

# Load sensitivity scores
sensitivity_scores = load_sensitivity_scores("sensitivity_scores.json")

# log_and_print("Model and tokenizer loaded successfully.")
log_and_print("Starting evaluation before float 16 ")
# evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
# log_and_print("Evaluation completed.")

model.half()

# Free up memory before starting evaluation
torch.cuda.empty_cache()
gc.collect()

# Convert parameters with sensitivity scores below the threshold to float16
convert_parameters_below_threshold_to_8bit(model, sensitivity_scores, threshold=10)

# List model parameters after conversion
list_model_parameters(model, "before")

def flip_lsb_mantissa(weight, dtype):
    """
    Function to flip the lower 8 bits of the mantissa of a weight in float16.
    Only applies to float16.
    """
    if dtype == torch.float16:
        # Convert the float16 weight to its binary (16-bit) representation
        weight_as_int = np.float16(weight).view(np.int16)
        
        # Define a mask to isolate and flip the lower 8 bits of the mantissa (bit 0 to 7 in float16)
        mantissa_lsb_mask = 0x00FF  # Mask to flip lower 8 bits of mantissa for float16
        
        # Flip the lower 8 bits of the mantissa by XORing with the mask
        modified_weight_as_int = weight_as_int ^ mantissa_lsb_mask

        # Convert the modified integer representation back to float16
        modified_weight = np.int16(modified_weight_as_int).view(np.float16)

        return modified_weight
    else:
        # If dtype is not float16, return the weight as it is (no change)
        return weight



def flip_lsb_exponent(weight, dtype):
    """
    Function to flip the LSB of the exponent of a weight in either float32 or float16.
    """
    if dtype == torch.float32:
        # Convert the float32 weight to its binary (32-bit) representation
        weight_as_int = np.float32(weight).view(np.int32)
        
        # Define a mask to isolate and flip the LSB of the exponent (bit 23 in float32)
        exponent_lsb_mask = 0x00800000  # This corresponds to bit 23 for float32 (LSB of the exponent)
        
        # Flip the LSB of the exponent by XORing with the mask
        modified_weight_as_int = weight_as_int ^ exponent_lsb_mask

        # Convert the modified integer representation back to float32
        modified_weight = np.int32(modified_weight_as_int).view(np.float32)
        
    elif dtype == torch.float16:
        # Convert the float16 weight to its binary (16-bit) representation
        weight_as_int = np.float16(weight).view(np.int16)
        
        # Define a mask to isolate and flip the LSB of the exponent (bit 10 in float16)
        exponent_lsb_mask = 0x0400  # This corresponds to bit 10 for float16

        # Flip the LSB of the exponent by XORing with the mask
        modified_weight_as_int = weight_as_int ^ exponent_lsb_mask

        # Convert the modified integer representation back to float16
        modified_weight = np.int16(modified_weight_as_int).view(np.float16)

    return modified_weight

def inject_error_with_probability_and_log(model, data_loader, tokenizer, labels, flip_probability=0.5, log_file='evaluation_log.csv'):
    """
    Function to inject MSB errors into the model's weights based on a given probability,
    evaluate the model, and save the results including original and modified weights into a CSV file.
    """
    counter = 0
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header for the CSV file
        writer.writerow(['Layer Name', 'Weight Index', 'Original Weight', 'Modified Weight', 'Accuracy',
                         'Memory Allocated', 'Memory Reserved', 'Initial Memory Allocated', 'Initial Memory Reserved',
                         'Final Memory Allocated', 'Final Memory Reserved'])

        for name, param in model.named_parameters():
            log_and_print(f"Changing layer {name}")
            if 'weight' in name and param.dtype == torch.float16:  # Only focus on float16 weight tensors
                dtype = param.dtype
                log_and_print(f"Changing float16 weights {name}")
                # Detach the tensor from the graph and clone to avoid inplace modification issues
                param_flat = param.detach().clone().view(-1)  
                num_weights = param_flat.numel()
  
                # Flip the LSB for all float16 weights in this layer
                original_values = param_flat.clone()
                  
                for i in range(num_weights):
                    original_value = param_flat[i].item()
                    modified_value = flip_lsb_mantissa(original_value, dtype)
                    param_flat[i] = torch.tensor(modified_value, dtype=dtype)

                # Inject the modified weights and evaluate the model
                with torch.no_grad():
                    param.data.copy_(param_flat.view(param.shape))  # Update the whole layer with modified values

                # Evaluate the model and get the metrics
                eval_metrics = evaluate_mixed_precision_model(model, data_loader, tokenizer, labels)
                log_and_print(f"evaluation {eval_metrics} ")
                # Log the result to the CSV file
                counter = counter + 1
                log_and_print(f"Counter for layer {counter} ")
                log_and_print(f"Original {original_values[i].item()} Changed {param_flat[i].item()} layer {name}")
                for i in range(num_weights):
                    writer.writerow([
                        name, i, original_values[i].item(), param_flat[i].item(), eval_metrics[0],  # accuracy
                        eval_metrics[1],  # memory_allocated
                        eval_metrics[2],  # memory_reserved
                        'N/A',  # Initial_memory_allocated, not available
                        'N/A',  # Initial_memory_reserved, not available
                        'N/A',  # Final_memory_allocated, not available
                        'N/A'   # Final_memory_reserved, not available
                    ])

                # Restore the original weights
                with torch.no_grad():
                    param.data.copy_(original_values.view(param.shape))  # Restore the original weights

# Free up memory before starting evaluation
torch.cuda.empty_cache()
gc.collect()

inject_error_with_probability_and_log(model, data_loader, tokenizer, labels, flip_probability=1, log_file='evaluation_log__float8_e5m2_LSB_inject.csv')

# Optional: print a confirmation message
print("evaluation_results_float8_e5m2_errorfloat16.xlsx")