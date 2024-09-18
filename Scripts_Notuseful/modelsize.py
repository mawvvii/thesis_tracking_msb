
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader
import time

originalmodel = resnet20_cifar()

# Load the pre-trained model state_dict
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

MSBmodel = resnet20_cifar()

# Load the pre-trained model state_dict
state_dict_MSB = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/SmallerModels/modified_model.pth')


try:
    originalmodel.load_state_dict(state_dict)
    print(f"Model loaded successfully from {state_dict}.")
    MSBmodel.load_state_dict(state_dict_MSB)
except Exception as e:
    print(f"Error loading the model: {e}")
    raise


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'Non-Trainable Parameters: {total_params - trainable_params:,}')
    
    return total_params, trainable_params

# Usage
total_params, trainable_params = count_parameters(originalmodel)
total_params_msb, trainable_params_msb = count_parameters(MSBmodel)

print(f"original {total_params, trainable_params} msb {total_params_msb, trainable_params_msb}")

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert bytes to megabytes
    print(f'Model Size: {size_all_mb:.2f} MB')

    return size_all_mb

# Usage
model_size_mb = model_size(originalmodel)
model_size_mbs = model_size(MSBmodel)

print(f"original {model_size_mb} msb {model_size_mbs}")

# Evaluate both models
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Load the modified data
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)
# Assuming you have dataloaders setup for test data
accuracy_full = evaluate_model(originalmodel, test_loader)
accuracy_msb = evaluate_model(MSBmodel, test_loader)

print(f'Accuracy (Full Precision): {accuracy_full}%')
print(f'Accuracy (MSB Only): {accuracy_msb}%')

# Function to measure inference time
def measure_inference_time(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    end_time = time.time()
    print(f"Inference Time: {end_time - start_time:.2f} seconds")
    return end_time - start_time

# Function to validate model accuracy
def validate_model_accuracy(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Model Accuracy: {accuracy:.2f}%')
    return accuracy

# Function to measure memory usage (during inference)
def measure_memory_usage(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    torch.cuda.reset_max_memory_allocated()
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
    
    max_memory = torch.cuda.max_memory_allocated(device)
    print(f"Max Memory Usage: {max_memory / (1024**2):.2f} MB")
    return max_memory

print("original")
measure_memory_usage(originalmodel, test_loader, device='cuda')
validate_model_accuracy(originalmodel, test_loader, device='cuda')
measure_inference_time(originalmodel, test_loader, device='cuda')

print("msb")
measure_memory_usage(MSBmodel, test_loader, device='cuda')
validate_model_accuracy(MSBmodel, test_loader, device='cuda')
measure_inference_time(MSBmodel, test_loader, device='cuda')