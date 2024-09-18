import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader
import torch.nn.functional as F


# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the test data using the provided function
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = resnet20_cifar()

# Modify the first convolutional layer to accept a single-channel input
model.conv1 = torch.nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)

# Load the pre-trained model state_dict
state_dict = torch.load('/scratch/users/Maryam/L-DNQ/ResNet20/save_models/LDNQ_16.pth')

# Adjust the weight of the first convolutional layer to match the single-channel input
state_dict['conv1.weight'] = state_dict['conv1.weight']

model.load_state_dict(state_dict)
model.eval()

# Function to perform inference and collect actual vs predicted labels and top 3 probabilities
def evaluate_model(test_loader, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds = []
    all_labels = []
    all_top3_probs = []
    all_top3_classes = []
    num_layers_used_list = []

    with torch.no_grad():
        for data, labels in test_loader:
            labels = labels.to(device)
            num_layers_used = 1
            top1_prob = 0.0
            
            # Start with the red layer only
            input_data = data[:, 0:1, :, :].to(device)  # Red channel
            
            while num_layers_used <= 3:
                outputs = model(input_data)
                probs = F.softmax(outputs, dim=1)
                top1_prob, _ = torch.max(probs, 1)

                print(f'Layers used: {num_layers_used}, Top1 probability: {top1_prob.mean().item()}')

                if top1_prob.mean().item() >= 0.95 or num_layers_used == 3:
                    break
                else:
                    if num_layers_used == 1:
                        input_data = data[:, 0:2, :, :].to(device)  # Red + Green channels
                    elif num_layers_used == 2:
                        input_data = data.to(device)  # Red + Green + Blue channels
                    num_layers_used += 1

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top3_probs.extend(probs.cpu().numpy())
            all_top3_classes.extend(outputs.cpu().numpy())
            num_layers_used_list.extend([num_layers_used] * len(labels))

    return all_labels, all_preds, all_top3_probs, all_top3_classes, num_layers_used_list

# Perform inference
actual_labels, predicted_labels, top3_probs, top3_classes, num_layers_used_list = evaluate_model(test_loader, model)

# Prepare the data for CSV
data = []
for actual, predicted, probs, classes, num_layers_used in zip(actual_labels, predicted_labels, top3_probs, top3_classes, num_layers_used_list):
    top1_prob, top2_prob, top3_prob = probs[:3]
    top1_class, top2_class, top3_class = [class_names[int(idx)] for idx in range(3)]
    data.append([
        class_names[actual],
        class_names[predicted],
        top1_prob,
        top1_class,
        top2_prob,
        top2_class,
        top3_prob,
        top3_class,
        num_layers_used
    ])

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=[
    'Actual', 'Predicted', 'Top1_Prob', 'Top1_Class', 'Top2_Prob', 'Top2_Class', 'Top3_Prob', 'Top3_Class', 'Num_Layers_Used'
])
df.to_csv('multi_layer_results.csv', index=False)

# Create a confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels, labels=range(len(class_names)))

# Save the confusion matrix to a text file
with open('confusion_matrix_LDNQ.txt', 'w') as f:
    f.write('Confusion Matrix\n')
    f.write('\t' + '\t'.join(class_names) + '\n')
    for i, row in enumerate(cm):
        f.write(class_names[i] + '\t' + '\t'.join(map(str, row)) + '\n')

print("Results saved to multi_layer_results.csv and confusion_matrix.txt")