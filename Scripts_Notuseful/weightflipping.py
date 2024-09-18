import torch
import numpy as np
import pandas as pd

def flip_bits_in_tensor(tensor, flip_probability):
    tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy iteration
    num_elements = tensor_flat.numel()
    
    # Determine the number of elements to flip based on the flip_probability
    num_flips = int(num_elements * flip_probability)
    
    # Get the indices to flip
    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    
    for idx in flip_indices:
        original_value = tensor_flat[idx].item()
        
        # Convert the float to a 32-bit binary string
        binary_str = np.binary_repr(np.float32(original_value).view(np.int32), width=32)
        
        # Choose a random bit to flip
        bit_to_flip = np.random.randint(32)
        
        # Flip the chosen bit
        flipped_binary_str = list(binary_str)
        flipped_binary_str[bit_to_flip] = '0' if flipped_binary_str[bit_to_flip] == '1' else '1'
        
        # Convert the binary string back to a float
        flipped_value = np.int32(int(''.join(flipped_binary_str), 2)).view(np.float32)
        
        # Assign the flipped value back to the tensor
        tensor_flat[idx] = torch.tensor(flipped_value)

    return tensor.view(tensor.size())  # Reshape back to original shape

def flip_bits_in_tensor_lsb(tensor, flip_probability):
    tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy iteration
    num_elements = tensor_flat.numel()
    
    # Determine the number of elements to flip based on the flip_probability
    num_flips = int(num_elements * flip_probability)
    
    # Get the indices to flip
    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    
    for idx in flip_indices:
        original_value = tensor_flat[idx].item()
        
        # Convert the float to a 32-bit binary string
        binary_str = np.binary_repr(np.float32(original_value).view(np.int32), width=32)
        
        # Flip only the least significant bit (the last bit in the binary representation)
        flipped_binary_str = list(binary_str)
        lsb_index = 31  # Index for the least significant bit
        flipped_binary_str[lsb_index] = '0' if flipped_binary_str[lsb_index] == '1' else '1'
        
        # Convert the binary string back to a float
        flipped_value = np.int32(int(''.join(flipped_binary_str), 2)).view(np.float32)
        
        # Assign the flipped value back to the tensor
        tensor_flat[idx] = torch.tensor(flipped_value)

    return tensor.view(tensor.size())  # Reshape back to original shape

# Function to inject errors into a state_dict
def inject_errors_into_state_dict_lsb(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            # Apply bit flipping to the tensor
            corrupted_tensor = flip_bits_in_tensor_lsb(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a floating-point tensor (e.g., integer), leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict


def inject_errors_into_state_dict(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            # Apply bit flipping to the tensor
            corrupted_tensor = flip_bits_in_tensor(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a floating-point tensor (e.g., integer), leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict


import torch
import numpy as np

def flip_random_lower_16_bit_in_tensor(tensor, flip_probability):
    tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy iteration
    num_elements = tensor_flat.numel()
    
    # Determine the number of elements to flip based on the flip_probability
    num_flips = int(num_elements * flip_probability)
    
    # Get the indices to flip
    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    
    for idx in flip_indices:
        original_value = tensor_flat[idx].item()
        
        # Convert the float to a 32-bit binary string
        binary_str = np.binary_repr(np.float32(original_value).view(np.int32), width=32)
        
        # Select a random bit to flip in the lower 16 bits (from index 16 to 31)
        random_bit_index = np.random.randint(16, 32)
        
        # Flip the selected bit
        flipped_binary_str = list(binary_str)
        flipped_binary_str[random_bit_index] = '0' if flipped_binary_str[random_bit_index] == '1' else '1'
        
        # Convert the binary string back to a float
        flipped_value = np.int32(int(''.join(flipped_binary_str), 2)).view(np.float32)
        
        # Assign the flipped value back to the tensor
        tensor_flat[idx] = torch.tensor(flipped_value)

    return tensor.view(tensor.size())  # Reshape back to original shape

# Function to inject errors into a state_dict
def inject_errors_into_state_dict_random_lower_16(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            # Apply bit flipping to the tensor
            corrupted_tensor = flip_random_lower_16_bit_in_tensor(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a floating-point tensor (e.g., integer), leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict

def flip_all_lower_16_bits_in_tensor(tensor, flip_probability):
    tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy iteration
    num_elements = tensor_flat.numel()
    
    # Determine the number of elements to flip based on the flip_probability
    num_flips = int(num_elements * flip_probability)
    print(f"number of flips {num_flips}")
    
    # Check if num_flips is greater than 0
    if num_flips == 0:
        print("No elements selected for flipping. Please adjust the flip_probability.")
        return tensor.view(tensor.size())  # Return tensor unchanged if no flips are made
    
    # Get the indices to flip
    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    
    # Store original and modified values for logging
    original_values = []
    modified_values = []
    
    for idx in flip_indices:
        original_value = tensor_flat[idx].item()
        
        # Convert the float to a 32-bit binary string
        binary_str = np.binary_repr(np.float32(original_value).view(np.int32), width=32)
        
        # Flip all 16 bits in the lower half (from index 16 to 31)
        flipped_binary_str = list(binary_str)
        for bit_index in range(16, 32):
            flipped_binary_str[bit_index] = '0' if flipped_binary_str[bit_index] == '1' else '1'
        
        # Convert the binary string back to a float
        flipped_value = np.int32(int(''.join(flipped_binary_str), 2)).view(np.float32)
        
        # Save original and modified values
        original_values.append(original_value)
        modified_values.append(flipped_value)
        
        # Assign the flipped value back to the tensor
        tensor_flat[idx] = torch.tensor(flipped_value)

    # Check if there are any values to save
    if len(original_values) == 0 or len(modified_values) == 0:
        print("No values were flipped, so no data to save to Excel.")
    else:
        # Create a DataFrame to save the original and modified values
        df = pd.DataFrame({
            'Original Values': original_values,
            'Modified Values': modified_values
        })
        
        # Save the DataFrame to an Excel file
        df.to_excel('weights_changes.xlsx', index=False)
        print("Weights changes have been saved to weights_changes.xlsx.")

    return tensor.view(tensor.size())  # Reshape back to original shape

# Function to inject errors into a state_dict and save changes
def inject_errors_into_state_dict_all_lower_16(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            # Apply bit flipping to the tensor and save changes
            corrupted_tensor = flip_all_lower_16_bits_in_tensor(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a floating-point tensor (e.g., integer), leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict


import torch
import numpy as np
import pandas as pd

def remove_lower_16_bits_in_tensor(tensor, flip_probability):
    tensor_flat = tensor.view(-1)  # Flatten the tensor to 1D for easy iteration
    num_elements = tensor_flat.numel()
    
    # Determine the number of elements to modify based on the flip_probability
    num_flips = max(1, int(num_elements * flip_probability))  # Ensure at least one element is modified
    
    # Print debug information
    print(f"Tensor has {num_elements} elements. Attempting to modify {num_flips} elements.")
    
    # Get the indices to modify
    flip_indices = np.random.choice(num_elements, num_flips, replace=False)
    
    # Store original and modified values for logging
    original_values = []
    modified_values = []
    
    for idx in flip_indices:
        original_value = tensor_flat[idx].item()
        
        # Convert the float to a 32-bit binary integer representation
        int_value = np.float32(original_value).view(np.int32)
        
        # Mask out (zero) the lower 16 bits using a bitwise AND with a mask
        masked_value = int_value & 0xFFFF0000
        
        # Convert the masked binary back to a float
        modified_value = np.int32(masked_value).view(np.float32)
        
        # Save original and modified values
        original_values.append(original_value)
        modified_values.append(modified_value)
        
        # Assign the modified value back to the tensor
        tensor_flat[idx] = torch.tensor(modified_value)

    # Check if there are any values to save
    if len(original_values) == 0 or len(modified_values) == 0:
        print("No values were modified, so no data to save to Excel.")
    else:
        # Create a DataFrame to save the original and modified values
        df = pd.DataFrame({
            'Original Values': original_values,
            'Modified Values': modified_values
        })
        
        # Save the DataFrame to an Excel file
        df.to_excel('weights_changes_removed_lower_bits.xlsx', index=False)
        print(f"Weights changes have been saved to weights_changes_removed_lower_bits.xlsx with {len(original_values)} changes.")

    return tensor.view(tensor.size())  # Reshape back to original shape

# Function to inject errors into a state_dict and save changes
def inject_errors_into_state_dict_remove_lower_16(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            # Apply lower 16 bits removal to the tensor and save changes
            print(f"Processing tensor: {key} with shape {tensor.shape}")
            corrupted_tensor = remove_lower_16_bits_in_tensor(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a floating-point tensor (e.g., integer), leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict
