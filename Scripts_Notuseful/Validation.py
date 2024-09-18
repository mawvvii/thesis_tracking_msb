import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function

# Define the transform to preprocess the input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to calculate accuracy
def accuracy(output, labels):
    _, preds = torch.max(output, 1)
    return torch.sum(preds == labels).item() / len(labels)

# Load the model
model = resnet20_cifar()
model.load_state_dict(torch.load('/scratch/users/Maryam/L-DNQ/ResNet20/ResNet20_pretrain.pth'))
print()
model.eval()

# Iterate through the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Helper function to forward pass through individual layers and print their output shapes
def forward_through_layers(model, data):
    x = data
    layer_shapes = {}

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    layer_shapes['conv1'] = x.shape

    for i, layer in enumerate(model.layer1):
        x = layer(x)
        layer_shapes[f'layer1_block{i}'] = x.shape

    for i, layer in enumerate(model.layer2):
        x = layer(x)
        layer_shapes[f'layer2_block{i}'] = x.shape

    for i, layer in enumerate(model.layer3):
        x = layer(x)
        layer_shapes[f'layer3_block{i}'] = x.shape

    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    layer_shapes['avgpool'] = x.shape

    x = model.fc(x)
    layer_shapes['fc'] = x.shape

    return x, layer_shapes

# Open a file to save the results
with open('Original_results.txt', 'w') as f:
    # Evaluate the model on the test set and print layer-wise shapes and final layer accuracy
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        output, layer_shapes = forward_through_layers(model, data)
        final_accuracy = accuracy(output, labels)
        f.write(f'Final layer accuracy: {final_accuracy:.4f}\n')
        for layer, shape in layer_shapes.items():
            f.write(f'Layer {layer} output shape: {shape}\n')

print("Results saved to results.txt")
