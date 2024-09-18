from models_CIFAR10.resnet import resnet20_cifar  # Import the model function
from utils.dataset import get_dataloader, get_modified_dataloader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from biasesflipping import inject_errors_into_biases
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define class names for CIFAR-10
class_names = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the modified data
test_loader = get_dataloader('CIFAR10', 'test', batch_size=100, shuffle=False)

# Load the model
model = resnet20_cifar()

# Load the pre-trained model state_dict
state_dict = torch.load('/scratch/users/Maryam/Thesis_Tracking/L-DNQ/ResNet20/ResNet20_pretrain.pth')

# Inject errors into the biases with a certain probability
flip_probability = 0.001  # For example, 10% of the biases will have bit flips
corrupted_state_dict = inject_errors_into_biases(state_dict, flip_probability)

# Load the corrupted state dict into the model
model.load_state_dict(corrupted_state_dict)
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
    all_top_probs_by_layer = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            labels = labels.to(device)
            input_data = data.to(device)  # Use the data directly, no need to manipulate channels
            
            print(f'Batch {batch_idx}: Input data shape: {input_data.shape}')
            
            outputs = model(input_data)
            probs = F.softmax(outputs, dim=1)
            max_probs, max_classes = torch.max(probs, 1)

            # Debugging: print top 3 probabilities and their classes
            top_probs, top_classes = torch.topk(probs, 3, dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get top 3 probabilities and their corresponding classes for each example
            all_top3_probs.extend(top_probs.cpu().numpy() * 100)  # Convert to percentage
            all_top3_classes.extend(top_classes.cpu().numpy())

            # Ensure each example has probabilities for 3 layers, filling with NaNs if necessary
            for i in range(len(labels)):
                all_top_probs_by_layer.append(probs[i].cpu().numpy())

    return all_labels, all_preds, all_top3_probs, all_top3_classes, all_top_probs_by_layer

# Perform inference
actual_labels, predicted_labels, top3_probs, top3_classes, top_probs_by_layer = evaluate_model(test_loader, model)


# Prepare the data for CSV
data = []
for idx, (actual, predicted, probs, classes) in enumerate(zip(actual_labels, predicted_labels, top3_probs, top3_classes)):
    top1_prob, top2_prob, top3_prob = probs
    print(f'{top1_prob} prob1 and prob2 {top2_prob}')
    top1_class, top2_class, top3_class = [class_names[idx] for idx in classes]

    data.append([
        class_names[actual],
        class_names[predicted],
        top1_prob,
        top1_class,
        top2_prob,
        top2_class,
        top3_prob,
        top3_class,

    ])

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=[
    'Actual', 'Predicted', 'Top1_Prob', 'Top1_Class', 'Top2_Prob', 'Top2_Class', 'Top3_Prob', 'Top3_Class'
])
df.to_excel('weight_directinefernce_error0001.xlsx', index=False)

cm = confusion_matrix(actual_labels, predicted_labels, labels=list(range(len(class_names))))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix 0001 Error biases')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix plot
plt.savefig('Confusion Matrix 0001 Error biases.png')
plt.show()