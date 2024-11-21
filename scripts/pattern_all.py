import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Phi3Config, Phi3ForCausalLM
import numpy as np
from prompt import prompts
import gc
from datasets import load_dataset
from collections import defaultdict
import sys

cache_strategy = 0
dump_csv = 0
num_layers = 32
num_channels = 3072

name_or_path = "microsoft/Phi-3.5-mini-instruct"
config = AutoConfig.from_pretrained(name_or_path, trust_remote_code = True, use_cache=True, caching_strategy=cache_strategy, dump_csv=dump_csv)                

tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code = True, config=config)
model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code = True, config=config, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.float16,)

def prompt_stats_new(past_kv, key_or_value=0):
     
    # Extract tensors for layer 0 
    tensor_first = past_kv[0][key_or_value] # (bsz, head, seqlen, headdim)
    bsz, heads, seqlen, headdim = tensor_first.shape
    tensor_first = tensor_first.transpose(1, 2).view(bsz, seqlen, heads * headdim).contiguous() # bsz, seqlen, hidden_size
    tensor_first = tensor_first[0, :, :] # (seqlen, hidden_size)
    
    hidden_size = tensor_first.size(1)
    # Initialize a dictionary to collect layer stats for each channel
    channel_stats_dict = {channel_idx: [] for channel_idx in range(hidden_size)}

    for layer in range(1, num_layers):
        tensor_next = past_kv[layer][key_or_value]
        tensor_next = tensor_next.transpose(1, 2).view(bsz, seqlen, heads * headdim).contiguous()
        tensor_next = tensor_next[0, :, :] # (seqlen, hidden_size)
        
        tensor_diff = (tensor_next - tensor_first).contiguous() # (seqlen, hidden_size)
        
        l2_norms = torch.linalg.vector_norm(tensor_diff, ord=2, dim=0) # (hidden_size,)
        
        assert l2_norms.shape == (hidden_size,), f"l2_norms must be of the shape ({hidden_size},)"
        
        average_changes = l2_norms / seqlen
        
        for channel_idx, avg_change in enumerate(average_changes):
            if avg_change < 0.01:
                channel_stats_dict[channel_idx].append((layer, avg_change.item()))

        del tensor_next, tensor_diff
        gc.collect()
        torch.cuda.empty_cache()
    
    # Convert dictionary to list of tuples (channel_idx, layer_list) for channels with results
    channel_stats = [(channel_idx, layer_list) for channel_idx, layer_list in channel_stats_dict.items() if layer_list]
            
    del tensor_first
    gc.collect()
    torch.cuda.empty_cache()
            
    return channel_stats


# this is what prompt_stats returns
# [
#     (0, [(1, 0.005), (4, 0.008)]),  # Channel 0 layers meeting the condition
#     (2, [(3, 0.009)])  # Channel 2 layers meeting the condition, etc.
# ]

dataset_path = "THUDM/LongBench"

ds = load_dataset(dataset_path, "2wikimqa", split="test") # size = 200
# ds = ds.train_test_split(test_size=100)
total_stats = []

i = 0
for prompt in ds:
    prompt = prompt["context"]
    sys.stdout.flush()
    i += 1
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda").input_ids
    
    # Check if length exceeds 10,000 and clip if necessary
    if input_ids.size(1) > 4000:
        input_ids = input_ids[:, :4000]  # Clip to the first 10,000 tokens
    print(f"prompt {i}: length {input_ids.shape[1]}")
    sys.stdout.flush()
    
    outputs = model(input_ids,
                    output_attentions=False, output_hidden_states=False, use_cache=True, return_dict=True)

    past_key_values = outputs.past_key_values # List(list(tensor)) tensor shape: (b,numheads, seqlen, headdim)
    # hidden_states=outputs.hidden_states # shape: (b, seqlen, embeddim)
    # attentions=outputs.attentions
    # logits = outputs.logits
    print(f"prefill done for prompt {i}")
    sys.stdout.flush()
    
    stats = prompt_stats_new(past_key_values, 0) # List of tuples (channels, layer_list). Layer_list = List of tuples (layer_idx, av change)
    print(f"stats computed for prompt {i}")
    sys.stdout.flush()

    del outputs, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    total_stats.append(stats) # List of "stats" for each prompt


def check_consistent_layers_across_prompts(total_stats):
    # total_stats is a list where each element is the output of `prompt_stats` for a given prompt
    num_prompts = len(total_stats)
    channel_layer_counts = defaultdict(lambda: defaultdict(int))  # {channel: {layer: count}}

    # Aggregate layers that meet the condition across prompts
    for prompt_stat in total_stats:
        for channel, layers in prompt_stat:
            for layer, avg_change in layers:
                channel_layer_counts[channel][layer] += 1

    # Determine consistent layers for each channel
    consistent_layers = {}
    for channel, layer_counts in channel_layer_counts.items():
        # Find layers that meet the condition in all prompts (or a majority of prompts)
        all_prompts_layers = [layer for layer, count in layer_counts.items() if count == num_prompts]
        majority_prompts_layers = [layer for layer, count in layer_counts.items() if count >= 0.8 * num_prompts]
        fifty_percent_prompts_layers = [layer for layer, count in layer_counts.items() if count >= 0.5 * num_prompts]
        
        consistent_layers[channel] = {
            "all_prompts": all_prompts_layers,
            "mostly_prompts": majority_prompts_layers,
            "fifty_prompts": fifty_percent_prompts_layers,
        }

    return consistent_layers  # {channel: {"all_prompts": [...], "mostly_prompts": [...]}}

def print_consistent_layers(consistent_layers_across_prompts):
    print("Consistent Layers Across Prompts\n" + "=" * 40)
    for channel, layers_info in consistent_layers_across_prompts.items():
        print(f"\CHANNEL {channel}:")
        
        all_prompts_layers = layers_info["all_prompts"]
        mostly_prompts_layers = layers_info["mostly_prompts"]
        fifty_prompts_layers = layers_info["fifty_prompts"]
        
        print("  Layers consistent across all prompts:")
        if all_prompts_layers:
            print("    ", ", ".join(map(str, all_prompts_layers)))
        else:
            print("    None")
        
        print("  Layers consistent in most prompts (80% or more):")
        if mostly_prompts_layers:
            print("    ", ", ".join(map(str, mostly_prompts_layers)))
        else:
            print("    None")
            
        print("  Layers consistent in atleast half of the prompts (50% or more):")
        if fifty_prompts_layers:
            print("    ", ", ".join(map(str, fifty_prompts_layers)))
        else:
            print("    None")
            

# prompt_stats() is called for each prompt and results are stored in total_stats list
# prompts_stats = [prompt_stats(past_kv_prompt_1), prompt_stats(past_kv_prompt_2), ...]
consistent_layers_across_prompts = check_consistent_layers_across_prompts(total_stats)

# Assuming consistent_layers_across_prompts has been calculated as shown previously
print_consistent_layers(consistent_layers_across_prompts)


# input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda").input_ids

# outputs = model(input_ids,
#                 output_attentions=True, output_hidden_states=True, use_cache=True, return_dict=True)

# past_key_values = outputs.past_key_values # List(list(tensor)) tensor shape: (b,numheads, seqlen, headdim)
# hidden_states=outputs.hidden_states # shape: (b, seqlen, embeddim)
# attentions=outputs.attentions
# logits = outputs.logits
    
# stats = prompt_stats(past_key_values, 1)
# gc.collect()
# torch.cuda.empty_cache()

# for stat in stats:
#     print("channel: ", stat[0], " num layers for it: ", len(stat[1]))