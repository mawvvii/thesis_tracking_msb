import torch
import numpy as np

# Function to flip bits in a tensor
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

# Function to inject errors into the biases of the state_dict
def inject_errors_into_biases(state_dict, flip_probability):
    corrupted_state_dict = {}
    
    for key, tensor in state_dict.items():
        if "bias" in key and torch.is_floating_point(tensor):
            # Apply bit flipping to the bias tensor
            corrupted_tensor = flip_bits_in_tensor(tensor.clone(), flip_probability)
            corrupted_state_dict[key] = corrupted_tensor
        else:
            # If it's not a bias tensor, leave it unchanged
            corrupted_state_dict[key] = tensor
    
    return corrupted_state_dict