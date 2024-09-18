import random
import os
import numpy as np
import torch
from torchvision import datasets, transforms

def flip_random_bit(byte_array):
    byte_idx = random.randint(0, len(byte_array) - 1)
    bit_position = random.randint(0, 7)
    byte_array[byte_idx] ^= (1 << bit_position)
    return byte_array

def modify_and_save_dataset(data_dir, output_dir, dataset_name='CIFAR10', flip_probability=1):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

        modified_data = []
        for i, (img, label) in enumerate(dataset):
            img_np = img.numpy()  # Convert image to numpy array
            
            # Decide whether to flip a bit in this image
            if random.random() < flip_probability:
                img_bytes = img_np.tobytes()  # Convert numpy array to bytes
                
                # Flip a random bit in the byte array
                modified_bytes = bytearray(img_bytes)
                modified_bytes = flip_random_bit(modified_bytes)
                
                # Convert bytes back to numpy array
                modified_img = np.frombuffer(modified_bytes, dtype=img_np.dtype).reshape(img_np.shape)
            else:
                # No modification, use the original image
                modified_img = img_np
            
            modified_data.append((torch.tensor(modified_img), label))
        
        # Save the modified dataset as a .pt file in the output directory
        os.makedirs(output_dir, exist_ok=True)
        modified_data_file = os.path.join(output_dir, f'{dataset_name}_modified100.pt')
        torch.save(modified_data, modified_data_file)
        print(f'Modified dataset saved to {modified_data_file}')

# Example usage:
data_dir = '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/data/CIFAR10'
output_dir = '/scratch/users/Maryam/Thesis_Tracking/L-DNQ/CIFAR10_modified100'
modify_and_save_dataset(data_dir, output_dir, dataset_name='CIFAR10', flip_probability=1)