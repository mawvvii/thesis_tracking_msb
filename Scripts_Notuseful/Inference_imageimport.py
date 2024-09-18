import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

# Define the transform to preprocess the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Define the inverse transform to convert normalized tensor back to image
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
                         std=[1/0.2023, 1/0.1994, 1/0.2010]),
    transforms.ToPILImage()
])

# Load the CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Choose a random index
random_idx = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_idx]

# Define class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Save the image as a PNG file
image_pil = inverse_transform(image)  # Convert tensor to PIL image
image_pil.save('random_cifar10_image.png')
print(f'Image saved as random_cifar10_image.png')

# Print the label
print(f'Label: {classes[label]}')
