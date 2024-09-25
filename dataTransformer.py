import torchvision.transforms as transforms

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the required input size for ViT
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet
])

from torchvision.datasets import CIFAR10

# Download the dataset and apply the transformations
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
