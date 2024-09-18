import torch
import torch.nn as nn
import numpy as np
import shap
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader
import matplotlib.pyplot as plt

# Load the pre-trained model
model = resnet20_cifar()
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

try:
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from the specified path.")
except Exception as e:
    print(f"Error loading the model: {e}")
    raise

# Set model to evaluation mode
model.eval()

# Move the model to the appropriate device if needed (e.g., CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Sample some data to use for SHAP explanations
# Convert it to a format compatible with SHAP (numpy array)
for images, labels in test_loader:
    sample_data = images.to(device)  # Ensure the data is on the same device as the model
    break  # Use the first batch as a sample set

# Use only a small sample for the SHAP background (e.g., first 100 samples)
background = sample_data[:100]

# Initialize SHAP Gradient Explainer with the model and background data
explainer = shap.GradientExplainer(model, background)

# Compute Shapley values for a subset of test data
shap_values = explainer.shap_values(sample_data[:10])  # Explaining the first 10 samples

# Convert data back to CPU for visualization if needed
shap_values = [sv.cpu().numpy() for sv in shap_values]
sample_data_np = sample_data[:10].cpu().numpy()

# Visualize the Shapley values for the first explained instance
shap.summary_plot(shap_values, sample_data_np, plot_type="bar")
