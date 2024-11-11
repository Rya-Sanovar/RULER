import os
import pandas as pd
import numpy as np

# Define paths and suffix length
cache_dir = "cache"
normal_dir = "normal"
suffix_len = 10
pref_len = 9
total_len = 19
bsz = 1
num_heads = 32
num_layers = 32 
atol = 1e-3

# Function to load and slice the normal CSV file based on suffix length
def load_and_slice(file_path, pref_len):
    data = pd.read_csv(file_path)
    return data.iloc[bsz * pref_len:].reset_index(drop=True)

# Function to load normal data and extract the suffix part for each head
def load_and_slice_query(file_path, num_heads, total_len, suffix_len):
    data = pd.read_csv(file_path)
    suffix_data = []
    
    # For each head, extract the suffix_len portion (last suffix_len rows for that head)
    for head_idx in range(num_heads):
        # Calculate the range of rows for this head
        start_idx = head_idx * total_len
        end_idx = start_idx + total_len
        
        # Extract the suffix part for this head (last suffix_len rows)
        head_suffix_data = data.iloc[end_idx - suffix_len:end_idx]
        suffix_data.append(head_suffix_data)
    
    # Combine all heads' suffix data back into a single DataFrame
    return pd.concat(suffix_data, axis=0).reset_index(drop=True)

# List of file names to compare
files_to_compare = ["layernorm1.csv", "preROPE_query.csv", "preROPE_key.csv", "preROPE_value.csv", "query.csv", 
                    "key.csv", "value.csv", "attn_output.csv", "out_proj.csv", "residattn.csv", 
                    "postattn_layernorm.csv", "mlp.csv", "mlpdropout.csv"]

# Function to compare files between cache and normal for a given layer
def compare_files_for_layer(layer_idx):
    print(f"Comparing files for layer {layer_idx}...")
    
    # Check files for this layer in both directories
    for file_name in files_to_compare:
        cache_path = f"{cache_dir}/layer_{layer_idx}/{file_name}"
        normal_path = f"{normal_dir}/layer_{layer_idx}/{file_name}"
        
        if os.path.exists(cache_path) and os.path.exists(normal_path):
            cache_data = pd.read_csv(cache_path)
            
            if file_name == "key.csv" or file_name == "value.csv":
                normal_data = pd.read_csv(normal_path)
            elif file_name == "query.csv" or file_name == "preROPE_query.csv" or file_name == "preROPE_key.csv" or file_name == "preROPE_value.csv":
                normal_data = load_and_slice_query(normal_path, num_heads=num_heads, total_len=total_len, suffix_len=suffix_len)
            else:
                normal_data = load_and_slice(normal_path, pref_len)
            
            # Ensure both dataframes have the same shape for comparison
            if cache_data.shape != normal_data.shape:
                print(f"Shape mismatch in file {file_name} for layer {layer_idx}")
                print(f"Cache Shape: {cache_data.shape}, Normal Shape: {normal_data.shape}")
                return True  # Stop at the first mismatch

            # Compare data values
            are_equal = np.allclose(cache_data.values, normal_data.values, atol=atol)
            if not are_equal:
                print(f"Mismatch found in file {file_name} for layer {layer_idx}")
                
                differing_indices = np.where(~np.isclose(cache_data.values, normal_data.values, atol=atol))

                # Loop through each differing index and print the differing values
                for idx in range(len(differing_indices[0])):
                    row_idx = differing_indices[0][idx]  # Get the row index
                    col_idx = differing_indices[1][idx]  # Get the column index
                    cache_value = cache_data.iloc[row_idx, col_idx]  # Get the value from cache
                    normal_value = normal_data.iloc[row_idx, col_idx]  # Get the value from normal
                    
                    print(f"Differing at row {row_idx}, column {col_idx}:")
                    print(f"  Cache Value: {cache_value}")
                    print(f"  Normal Value: {normal_value}")
                
                return True  # Stop at the first mismatch
    return False  # No mismatch in this layer

# Function to compare all layers until a mismatch is found
def compare_all_layers():
    for layer_idx in range(num_layers):
        mismatch_found = compare_files_for_layer(layer_idx)
        if mismatch_found:
            print(f"First mismatch found in layer {layer_idx}")
            break
    else:
        print("No mismatches found across all layers")

# Compare all layers
compare_all_layers()
