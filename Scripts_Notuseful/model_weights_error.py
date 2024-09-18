from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader, get_modified_dataloader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from weightflipping import inject_errors_into_state_dict
import pandas as pd
import numpy as np

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the modified data
# test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)
test_loader = get_modified_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = resnet20_cifar()

# Load the original state dict
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

# Inject errors with a certain probability
flip_probability = 0.01  # For example, 10% of the weights will have bit flips
corrupted_state_dict = inject_errors_into_state_dict(state_dict, flip_probability)

# Load the corrupted state dict into the model
model.load_state_dict(corrupted_state_dict)
model.eval()

# Function to dynamically modify the first convolutional layer based on the number of input channels
def adjust_model_conv1(model, num_input_channels, device):
    print(f'Adjusting model to have {num_input_channels} input channels.')
    # Create a new convolutional layer with the required number of input channels
    old_conv1_weight = model.conv1.weight.data.clone()  # Save the old weights
    new_conv1 = torch.nn.Conv2d(num_input_channels, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    
    # Adjust the weights of the new convolutional layer
    with torch.no_grad():
        if num_input_channels == 1:
            new_conv1.weight.data = old_conv1_weight.mean(dim=1, keepdim=True).to(device)
        elif num_input_channels == 2:
            new_conv1.weight.data[:, :2, :, :] = old_conv1_weight[:, :2, :, :].to(device)
        else:  # num_input_channels == 3
            new_conv1.weight.data[:, :2, :, :] = old_conv1_weight[:, :2, :, :].to(device)
            new_conv1.weight.data[:, 2:, :, :] = old_conv1_weight[:, 1:2, :, :].expand(-1, 1, -1, -1).to(device)  # Initialize the third channel with the second channel's weights
    
    # Replace the old convolutional layer with the new one
    model.conv1 = new_conv1
    print(f'New conv1 layer: {model.conv1}')

# Function to perform inference and collect actual vs predicted labels and top 3 probabilities
def evaluate_model(test_loader, model, threshold):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []
    all_top3_probs = []
    all_top3_classes = []
    num_layers_used_list = []
    all_top_probs_by_layer = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            labels = labels.to(device)
            num_layers_used = torch.ones(len(labels), device=device)  # Track layers used for each example
            
            # Start with the red layer only
            input_data = data[:, 0:1, :, :].to(device)  # Red channel
            input_data_2ch = torch.zeros_like(data[:, :2, :, :]).to(device)
            input_data_3ch = torch.zeros_like(data).to(device)
            print(f'Batch {batch_idx}: Initial input data shape: {input_data.shape}')
            adjust_model_conv1(model, 1, device)
            
            batch_layer_probs = [[] for _ in range(len(labels))]  # Track top probabilities for each example in the batch
            stop_inference = torch.zeros(len(labels), dtype=torch.bool, device=device)  # Stop inference for each example

            while num_layers_used.max() <= 3 and not torch.all(stop_inference):
                print(f'Batch {batch_idx}: Forward pass with {num_layers_used.max().item()} channel(s).')
                outputs = model(input_data)
                probs = F.softmax(outputs, dim=1)
                max_probs, max_classes = torch.max(probs, 1)

                # Debugging: print top 3 probabilities and their classes
                top_probs, top_classes = torch.topk(probs, 3, dim=1)
                # for i in range(len(labels)):
                #     # print(f'Example {i}, Layer {num_layers_used[i].item()}, Top 3 classes: {[class_names[c] for c in top_classes[i]]}, Top 3 probs: {top_probs[i]}')

                # Track top probabilities for the current layer for each example in the batch
                for i in range(len(labels)):
                    batch_layer_probs[i].append(probs[i].cpu().numpy())

                # Update stop inference condition for each example
                stop_inference = stop_inference | (max_probs >= threshold)
                
                # Determine which examples still need more layers and update input_data accordingly
                if not torch.all(stop_inference):
                    if num_layers_used.max() == 1:
                        input_data_2ch[:, :1, :, :] = data[:, :1, :, :]
                        input_data_2ch[:, 1:, :, :] = data[:, 1:2, :, :]
                        input_data = input_data_2ch
                        adjust_model_conv1(model, 2, device)
                    elif num_layers_used.max() == 2:
                        input_data_3ch[:, :2, :, :] = data[:, :2, :, :]
                        input_data_3ch[:, 2:, :, :] = data[:, 2:3, :, :]
                        input_data = input_data_3ch
                        adjust_model_conv1(model, 3, device)

                    num_layers_used[~stop_inference] += 1

            print(f'Batch {batch_idx}: Final input data shape: {input_data.shape}')
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get top 3 probabilities and their corresponding classes for each example
            all_top3_probs.extend(top_probs.cpu().numpy() * 100)  # Convert to percentage
            all_top3_classes.extend(top_classes.cpu().numpy())

            # Ensure each example has probabilities for 3 layers, filling with NaNs if necessary
            for i in range(len(labels)):
                while len(batch_layer_probs[i]) < 3:
                    batch_layer_probs[i].append(np.nan * np.ones_like(batch_layer_probs[i][0]))
                all_top_probs_by_layer.extend(batch_layer_probs[i])

            num_layers_used_list.extend(num_layers_used.cpu().numpy())

    return all_labels, all_preds, all_top3_probs, all_top3_classes, all_top_probs_by_layer, num_layers_used_list

thresholds = [0.99, 0.98, 0.92, 0.87, 0.85]

# Initialize a dictionary to hold dataframes for each threshold
dfs = {}

for threshold in thresholds:
    # Perform inference
    actual_labels, predicted_labels, top3_probs, top3_classes, top_probs_by_layer, num_layers_used_list = evaluate_model(test_loader, model, threshold)
    
    # Prepare the data for CSV
    data = []
    for idx, (actual, predicted, probs, classes, num_layers_used) in enumerate(zip(actual_labels, predicted_labels, top3_probs, top3_classes, num_layers_used_list)):
        top1_prob, top2_prob, top3_prob = probs
        top1_class, top2_class, top3_class = [class_names[idx] for idx in classes]
        
        # Collect top probabilities from each layer
        try:
            top_prob_layer1 = np.nanmax(top_probs_by_layer[idx * 3]) * 100 if num_layers_used >= 1 else np.nan
            top_prob_layer2 = np.nanmax(top_probs_by_layer[idx * 3 + 1]) * 100 if num_layers_used >= 2 else np.nan
            top_prob_layer3 = np.nanmax(top_probs_by_layer[idx * 3 + 2]) * 100 if num_layers_used >= 3 else np.nan
        except IndexError as e:
            print(f'IndexError at idx {idx}: {e}')
            print(f'top_probs_by_layer length: {len(top_probs_by_layer)}')
            print(f'Expected length for 3 * idx: {3 * idx}')
            continue

        data.append([
            class_names[actual],
            class_names[predicted],
            top1_prob,
            top1_class,
            top2_prob,
            top2_class,
            top3_prob,
            top3_class,
            num_layers_used,
            top_prob_layer1,
            top_prob_layer2,
            top_prob_layer3
        ])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=[
        'Actual', 'Predicted', 'Top1_Prob', 'Top1_Class', 'Top2_Prob', 'Top2_Class', 'Top3_Prob', 'Top3_Class', 'Num_Layers_Used', 'Top_Prob_Layer1', 'Top_Prob_Layer2', 'Top_Prob_Layer3'
    ])
    
    # Add dataframe to the dictionary
    dfs[f'Threshold_{int(threshold * 100)}'] = df

# Save all dataframes to a single Excel file with multiple sheets
with pd.ExcelWriter('weight_error_fullmodel_1percent.xlsx', engine='xlsxwriter') as writer:
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print('sheetname {sheet_name}')



