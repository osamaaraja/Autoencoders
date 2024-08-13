
import numpy as np
import torch

############################################ sample data creation ############################################

array = np.linspace(1,100,100)
print(array)
print("-"*50)

# hyperparameters

window_length = 20
jump = window_length //2  # set as needed to make jumps from one window to another without overlap
stride = 1 # set as needed for the shift of window in ovelapping scenario
batch_dim = 2

############################################ creating windowed data ############################################

chunk = []
chunk_nonoverlap = []
# Non overlapping windows 
for idx in range(0, len(array), jump):
    start = idx
    stop = start + jump
    
    # Ensure the stop index does not go out of bounds
    if stop <= len(array):
        chunk = array[start:stop]
        print("start", start)
        print("stop", stop)
        print("chunk", chunk)
        print("-" * 50)

        chunk_nonoverlap.append(chunk)

chunk_nonoverlap = np.array(chunk_nonoverlap)
print("Final non-overlapping chunk shape", chunk_nonoverlap.shape)

chunk = []
chunk_overlap = []

############################################ Overalapping windows ############################################
for idx in range(0, len(array) - window_length + 1, stride):
    start = idx
    stop = start + window_length
    
    # Ensure the stop index does not go out of bounds
    if stop <= len(array):
        chunk = array[start:stop]
        print("start", start)
        print("stop", stop)
        print("chunk", chunk)
        print("-" * 50)

        if stop == len(array) and len(chunk_overlap)%2 == 0:
            break

        else:
            chunk_overlap.append(chunk)
    
            
chunk_overlap = np.array(chunk_overlap)

####################### converting to torch and adding extra dimension to the tensor #########################

tensor_nonoverlap = torch.tensor(chunk_nonoverlap)
print(f"original shape of non-overlapping data tensor: {tensor_nonoverlap.shape}")
tensor_nonoverlap = tensor_nonoverlap.unsqueeze(1)
print(f"adding dimension to the non-overlapping tensor: {tensor_nonoverlap.shape}")


tensor_overlap = torch.tensor(chunk_overlap)
print(f"original shape of overlapping data tensor: {tensor_overlap.shape}")
tensor_overlap = tensor_overlap.unsqueeze(1)
print(f"adding dimension to the overlapping tensor: {tensor_overlap.shape}")

##################################  setting the batch dimension of the tensor #################################

tensor_nonoverlap_batched = tensor_nonoverlap.view(batch_dim, -1, tensor_nonoverlap.size(2))
print(f"tensor_nonoverlap_batched.shape: {tensor_nonoverlap_batched.shape}")

tensor_overlap_batched = tensor_overlap.view(batch_dim, -1, tensor_overlap.size(2))
print(f"tensor_overlap_batched.shape: {tensor_overlap_batched.shape}")







