import torch
from torchvision import transforms
from PIL import Image
import json
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function

# Define the transform to preprocess the input data
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the model
model = resnet20_cifar()
model.load_state_dict(torch.load('/scratch/users/Maryam/L-DNQ/ResNet20/save_models/LDNQ.pth'))
model.eval()

# Load class names
# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Function to preprocess and classify an image
def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    model.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class
    predicted_class = class_names[predicted.item()]
    return predicted_class

# Path to the downloaded image
image_path = '/scratch/users/Maryam/L-DNQ/random_cifar10_image.png'

# Classify the image
predicted_class = classify_image(image_path)
print(f'The predicted class for the given image is: {predicted_class}')
