import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Phi3Config, Phi3ForCausalLM
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from prompt import prompts
from scipy.stats import spearmanr
import gc
import sys

gc.collect()
torch.cuda.empty_cache()

cache_strategy = 0
name_or_path = "microsoft/Phi-3.5-mini-instruct"
config = AutoConfig.from_pretrained(name_or_path, trust_remote_code = True, use_cache=True, caching_strategy=cache_strategy)                

tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code = True, config=config)
model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code = True, config=config, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.float16,)

input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda").input_ids

print("length: ", input_ids.shape)

outputs = model(input_ids,
                output_attentions=True, output_hidden_states=True, use_cache=True, return_dict=True)

print("done")

past_key_values = outputs.past_key_values # List(list(tensor)) tensor shape: (b,numheads, seqlen, headdim)
hidden_states=outputs.hidden_states # shape: (b, seqlen, embeddim)
attentions=outputs.attentions
logits = outputs.logits

def plot_tensor_given_head_given_layer(past_kv, key_or_value=0, head_idx=0, layer_idx=0, output_dir="layer25_plots"):
    """
    Optimized 3D plot of either the key or value tensor.

    Args:
    - past_kv: list of (key, value) tensors
    - head_idx: index of the attention head
    - layer_idx: index of the layer
    - key_or_value: 0 for key, 1 for value
    - output_dir: directory to save plots
    """
    # Extract the specified tensor for the given layer and head
    tensor = past_kv[layer_idx][key_or_value]  # 0 for key, 1 for value
    tensor_type = "key" if key_or_value == 0 else "value"
    tensor = tensor[0, head_idx, :, :]  # (seqlen, headdim)
    
    # Convert tensor to numpy array for easier handling with matplotlib
    tensor_to_plot = tensor.transpose(0,1).detach().cpu().numpy()  # (headdim, seqlen)
    headdim, seqlen = tensor_to_plot.shape

    # Create meshgrid for plotting
    headdim_vals = np.arange(headdim)
    seqlen_vals = np.arange(seqlen)
    headdim_vals, seqlen_vals = np.meshgrid(headdim_vals, seqlen_vals, indexing='ij')

    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(headdim_vals, seqlen_vals, tensor_to_plot, cmap='hot' if tensor_type == "key" else 'cool', edgecolor='none')

    # Add a color bar
    plt.colorbar(surf, ax=ax, label=f"{tensor_type.capitalize()} Magnitude")

    # Set plot titles and labels
    ax.set_title(f"{tensor_type.capitalize()} Tensor Magnitude for Layer {layer_idx}, Head {head_idx}", fontsize=10)
    ax.set_xlabel("Channel", fontsize=8, labelpad=10)
    ax.set_ylabel("Token", fontsize=8, labelpad=10)
    ax.set_zlabel("Magnitude", fontsize=8, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Save the plot with the appropriate file name
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"{tensor_type}_layer{layer_idx}_head{head_idx}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {tensor_type} plot as {plot_filename}")

    plt.show()
    plt.close(fig)

def plot_tensor_all_heads_given_layer(past_kv, key_or_value=0, layer_idx=0, output_dir="plots"):
    """
    Optimized 3D plot of either the key or value tensor.

    Args:
    - past_kv: list of (key, value) tensors
    - layer_idx: index of the layer
    - key_or_value: 0 for key, 1 for value
    - output_dir: directory to save plots
    """
    # Extract the specified tensor for the given layer and head
    tensor = past_kv[layer_idx][key_or_value]  # 0 for key, 1 for value
    tensor_type = "key" if key_or_value == 0 else "value"
    tensor = tensor[0, :, :, :]  # (head, seqlen, headdim)
    h, seqlen, headdim = tensor.shape
    # Convert tensor to numpy array for easier handling with matplotlib
    tensor = tensor.view(h, seqlen, headdim).transpose(0,1).view(seqlen, h, headdim).reshape(seqlen, h*headdim).contiguous()  # (seqlen, hiddendim)
    tensor = tensor.transpose(0,1) # (hiddendim, seqlen)
    tensor_to_plot = tensor.detach().cpu().numpy() 
    hiddim, seqlen = tensor_to_plot.shape

    # Create meshgrid for plotting
    hiddim_vals = np.arange(hiddim)
    seqlen_vals = np.arange(seqlen)
    hiddim_vals, seqlen_vals = np.meshgrid(hiddim_vals, seqlen_vals, indexing='ij')

    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(hiddim_vals, seqlen_vals, tensor_to_plot, cmap='hot' if tensor_type == "key" else 'cool', edgecolor='none')

    # Add a color bar
    plt.colorbar(surf, ax=ax, label=f"{tensor_type.capitalize()} Magnitude")

    # Set plot titles and labels
    ax.set_title(f"{tensor_type.capitalize()} Tensor Magnitude for Layer {layer_idx}, All heads", fontsize=10)
    ax.set_xlabel("Channel", fontsize=8, labelpad=10)
    ax.set_ylabel("Token", fontsize=8, labelpad=10)
    ax.set_zlabel("Magnitude", fontsize=8, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Save the plot with the appropriate file name
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"{tensor_type}_layer{layer_idx}_allheads.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {tensor_type} plot as {plot_filename}")

    plt.show()
    plt.close(fig)

def plot_tensor_difference_between_layers_given_head(past_kv, key_or_value=0, head_idx=0, layer_idx=0, output_dir="diff_plots"):
    """
    3D plot of the difference in magnitude between the specified layer and the next layer for a given head.
    Args:
    - past_kv: list of (key, value) tensors
    - key_or_value: 0 for key, 1 for value
    - head_idx: index of the attention head
    - layer_idx: index of the current layer
    - output_dir: directory to save plots
    """
    # Check that the next layer exists
    if layer_idx + 1 >= len(past_kv):
        print(f"Layer {layer_idx+1} does not exist in the provided data.")
        return
    
    tensor_type = "key" if key_or_value == 0 else "value"
    # Extract tensors for the current and next layer
    tensor_current = past_kv[layer_idx][key_or_value][0, head_idx, :, :]  # (seqlen, headdim)
    tensor_next = past_kv[layer_idx + 1][key_or_value][0, head_idx, :, :]  # (seqlen, headdim)
    
    tensor_diff = (tensor_next - tensor_current).contiguous().transpose(0,1) # headdim, seqlen
    headdim, seqlen = tensor_diff.shape
    
    # Compute the difference between the tensors
    tensor_diff = tensor_diff.detach().cpu().numpy()  # Convert to numpy for plotting
    
    # Create meshgrid for plotting
    headdim_vals = np.arange(headdim)
    seqlen_vals = np.arange(seqlen)
    headdim_vals, seqlen_vals = np.meshgrid(headdim_vals, seqlen_vals, indexing='ij')

    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface of the difference
    cmap_choice = 'RdBu' if key_or_value == 0 else 'coolwarm'  # Colormap choice for key or value
    surf = ax.plot_surface(headdim_vals, seqlen_vals, tensor_diff, cmap=cmap_choice, edgecolor='none')

    # Add a color bar
    plt.colorbar(surf, ax=ax, label=f"{tensor_type} Difference Magnitude")

    # Set plot titles and labels
    ax.set_title(f"Difference in {tensor_type.capitalize()} Magnitude Between Layers {layer_idx} and {layer_idx+1} for Head {head_idx}", fontsize=10)
    ax.set_xlabel("Channel", fontsize=8, labelpad=10)
    ax.set_ylabel("Token", fontsize=8, labelpad=10)
    ax.set_zlabel("Magnitude Difference", fontsize=8, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # Save the plot with the appropriate file name
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"{tensor_type}_diff_betn_layers{layer_idx}-{layer_idx+1}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {tensor_type} difference plot as {plot_filename}")

    plt.show()
    plt.close(fig)
    
    del tensor_diff
    gc.collect()
    torch.cuda.empty_cache()

def closest_channels_between_layers_given_head(past_kv, key_or_value=0, head_idx=0, layer_idx=0, output_dir="closest_channel_plots"):
    # Check that the next layer exists
    if layer_idx + 1 >= len(past_kv):
        print(f"Layer {layer_idx+1} does not exist in the provided data.")
        return
    
    tensor_type = "key" if key_or_value == 0 else "value"
    # Extract tensors for the current and next layer
    tensor_current = past_kv[layer_idx][key_or_value][0, head_idx, :, :]  # (seqlen, headdim)
    tensor_next = past_kv[layer_idx + 1][key_or_value][0, head_idx, :, :]  # (seqlen, headdim)
    
    tensor_diff = (tensor_next - tensor_current).contiguous()  # (seqlen, headdim)
    seqlen, headdim = tensor_diff.shape

    # List to hold channel info with Spearman correlation and L2 norm
    channel_stats = []

    for channel in range(headdim):
        # Extract the channel across all tokens (seqlen) for current and next layer tensors
        channel_current = tensor_current[:, channel].detach().cpu().numpy()
        channel_next = tensor_next[:, channel].detach().cpu().numpy()
        
        # Compute Spearman's rank correlation
        spearman_corr, _ = spearmanr(channel_current, channel_next)
        
        difference = tensor_diff[:, channel]
        
        # Compute L2 norm of the difference in this channel
        l2_norm = torch.norm( difference, p=2).item()
        
        average_change = l2_norm / difference.numel()
        
        # Append the channel number, correlation, and L2 norm to the list
        channel_stats.append((channel, spearman_corr, l2_norm, average_change))

    # Sort the list by Spearman correlation in descending order
    channel_stats = sorted(channel_stats, key=lambda x: x[3], reverse=False)

    # Print the results

    lmao = 0
    for channel, spearman_corr, l2_norm, average_change in channel_stats:
        if average_change < 0.01:
            # print(f"Channel {channel}: Spearman Correlation = {spearman_corr:.4f}, L2 Norm = {l2_norm:.4f}, average change = {average_change:.4f}")
            lmao += 1
    if (lmao > 0):
        print(f"#### Channel statistics for {tensor_type} tensor at head {head_idx}, layer: {layer_idx} - {layer_idx + 1}: ####")
        print(f"Number of channels with average change < 0.01 = {lmao}")

def plot_tensor_difference_between_heads_within_layer(past_kv, key_or_value=0, layer_idx=0, output_dir="head_diff_plots"):
    """
    Plots the difference in magnitude between tensors of every pair of heads within the specified layer.
    Args:
    - past_kv: list of (key, value) tensors
    - key_or_value: 0 for key, 1 for value
    - layer_idx: index of the layer to analyze
    - output_dir: directory to save plots
    """
    tensor_type = "key" if key_or_value == 0 else "value"

    # Check that the layer exists
    if layer_idx >= len(past_kv):
        print(f"Layer {layer_idx} does not exist in the provided data.")
        return

    # Extract the tensor for the specified layer
    layer_tensor = past_kv[layer_idx][key_or_value][0]  # Shape: (num_heads, seqlen, headdim)
    num_heads, seqlen, headdim = layer_tensor.shape

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through every unique pair of heads (x, y)
    for head_x in range(num_heads):
        for head_y in range(head_x + 1, num_heads):  # Only consider each unique pair once

            # Compute the difference in magnitude between the tensors of head_x and head_y
            tensor_diff = (layer_tensor[head_x, :, :] - layer_tensor[head_y, :, :]).transpose(0,1).contiguous()  # Shape: (headdim, seqlen)

            # detach and move to cpu for plotting
            tensor_diff = tensor_diff.detach().cpu().numpy()
            
            # Create meshgrid for plotting
            headdim_vals = np.arange(headdim)
            seqlen_vals = np.arange(seqlen)
            headdim_vals, seqlen_vals = np.meshgrid(headdim_vals, seqlen_vals, indexing='ij')

            # Create the figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface of the difference
            cmap_choice = 'RdBu' if key_or_value == 0 else 'coolwarm'  # Colormap choice for key or value
            surf = ax.plot_surface(headdim_vals, seqlen_vals, tensor_diff, cmap=cmap_choice, edgecolor='none')

            # Add a color bar
            plt.colorbar(surf, ax=ax, label=f"{tensor_type} Difference Magnitude")

            # Set plot titles and labels
            ax.set_title(f"Difference in {tensor_type.capitalize()} Magnitude Between Heads {head_x} and {head_y} in Layer {layer_idx}", fontsize=10)
            ax.set_xlabel("Channel", fontsize=8, labelpad=10)
            ax.set_ylabel("Token", fontsize=8, labelpad=10)
            ax.set_zlabel("Magnitude Difference", fontsize=8, labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=6)

            # Save the plot with the appropriate file name
            plot_filename = os.path.join(output_dir, f"{tensor_type}_diff_head{head_x}-{head_y}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved {tensor_type} difference plot between head {head_x} and head {head_y} as {plot_filename}")

            # Show and close the figure to free memory
            plt.show()
            plt.close(fig)

    # Clean up to free memory
    del layer_tensor, tensor_diff
    gc.collect()
    torch.cuda.empty_cache()

heads = 32
layers = 32

# for layer_idx in range(layers):
#     plot_tensor_difference_between_heads_within_layer(past_key_values, key_or_value=0, layer_idx=layer_idx, output_dir=f"head_diff_for_layer{layer_idx}_plot")
#     plot_tensor_difference_between_heads_within_layer(past_key_values, key_or_value=1, layer_idx=layer_idx, output_dir=f"head_diff_for_layer{layer_idx}_plot")

for h in range(heads):
    for key_or_value in [0,1]:
        tensor_type = "KEY" if key_or_value == 0 else "VALUE"
        print(f"###### HEAD {h} : {tensor_type} ######")
        for l in range(layers-1):
            closest_channels_between_layers_given_head(past_key_values, key_or_value=key_or_value, head_idx=h, layer_idx=l)

# for h in range(heads):
#     plot_tensor_given_head_given_layer(past_key_values, 0, h, 25)
#     plot_tensor_given_head_given_layer(past_key_values, 1, h, 25)
    
# for l in range(1):
#     plot_tensor_all_heads_given_layer(past_key_values, 0, l, output_dir="allhead_plots")
#     plot_tensor_all_heads_given_layer(past_key_values, 1, l, output_dir="allhead_plots")

# for h in range(heads):
#     for l in range(layers-1):
#         plot_tensor_difference_between_layers_given_head(past_key_values, 0, h, l, f"diff_plots_head{h}")
#         plot_tensor_difference_between_layers_given_head(past_key_values, 1, h, l, f"diff_plots_head{h}")
