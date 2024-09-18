import torch
from torchsummary import summary
from torchviz import make_dot
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from torch.autograd import Variable
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import graphviz
import torch.nn as nn

model = resnet20_cifar()

state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/save_models/LDNQ_16.pth')

# Load the state_dict into the model
model.load_state_dict(state_dict)

# Print out the model architecture to see all layers
for name, layer in model.named_modules():
    print(f"Layer Name: {name}, Layer Type: {layer}")

essential_layers = nn.Sequential()
non_essential_layers = nn.Sequential()

for name, layer in model.named_children():
    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        essential_layers.add_module(name, layer)
    elif isinstance(layer, (nn.Dropout, nn.MaxPool2d, nn.ReLU)):
        non_essential_layers.add_module(name, layer)
    else:
        # If the layer is a Sequential or any custom module, you might need to inspect it further
        # Here we assume that sequential blocks are essential by default
        essential_layers.add_module(name, layer)

# Save to text file
with open('essential_layers_16.txt', 'w') as f:
    f.write("Essential Layers:\n")
    f.write(str(essential_layers) + "\n\n")

with open('non_essential_layers_16.txt', 'w') as f:
    f.write("Non-Essential Layers:\n")
    f.write(str(non_essential_layers) + "\n\n")


save_directory = '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/SmallerModels'
save_path_essential = f'{save_directory}/essential_layers_model_16.pth'
save_path_nonessential = f'{save_directory}/non_essential_layers_model_16.pth'

# Save the essential layers to the specified .pth file
torch.save(essential_layers.state_dict(), save_path_essential)
torch.save(non_essential_layers.state_dict(), save_path_nonessential)

print("Essential layers saved as essential_layers_model.pth")


# print("Essential Layers:")
# print(essential_layers)

# print("\nNon-Essential Layers:")
# print(non_essential_layers)



state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

# Load the state_dict into the model
model.load_state_dict(state_dict)

print(model)

# Provide the correct input size
summary(model, input_size=(2, 3, 32, 32))

# # Print all the keys in the state_dict
# print("State Dictionary Keys:")
# for key in state_dict.keys():
#     print(key)

# # Load the state_dict into the model
# model = resnet20_cifar()  # Adjust according to your model definition
# model.load_state_dict(state_dict)

# # Print the model architecture
# print("\nModel Architecture:")
# print(model)

# # Extract layers and their parameter shapes
# layers = {}
# for key, tensor in state_dict.items():
#     layer = key.split('.')[0]
#     param = key.split('.')[1]
#     if layer not in layers:
#         layers[layer] = []
#     layers[layer].append((param, tensor.shape))

# # Plotting
# fig, ax = plt.subplots(figsize=(10, len(layers) * 0.6))
# y = range(len(layers))
# ax.barh(y, [sum([torch.prod(torch.tensor(shape)) for _, shape in params]) for params in layers.values()])
# ax.set_yticks(y)
# ax.set_yticklabels(layers.keys())
# ax.set_xlabel('Number of Parameters')
# ax.set_title('Model Layers and Parameter Counts')

# plt.tight_layout()
# plt.show()

