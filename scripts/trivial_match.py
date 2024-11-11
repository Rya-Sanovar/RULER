# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

dump_csv = 0

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Phi3Config, Phi3ForCausalLM

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.cache_strategy = self.generation_kwargs.pop('caching_strategy')
        self.dump_csv = self.generation_kwargs.pop('dump_csv')
        
        self.config = AutoConfig.from_pretrained(name_or_path, trust_remote_code = True, use_cache=True, caching_strategy=self.cache_strategy, dump_csv=self.dump_csv,)                

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code = True, config=self.config)
        self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code = True, config=self.config, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.float16,)

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model(
            **inputs,
        )
        
        return outputs
    
llm1 = HuggingFaceModel(
            name_or_path="microsoft/Phi-3.5-mini-instruct",
            do_sample=False,
            repetition_penalty=1,
            temperature=1.0,
            top_p=1,
            top_k=0,
            stop="",
            max_new_tokens=1,
            caching_strategy=1,
            dump_csv=dump_csv,
        )

llm2 = HuggingFaceModel(
            name_or_path="microsoft/Phi-3.5-mini-instruct",
            do_sample=False,
            repetition_penalty=1,
            temperature=1.0,
            top_p=1,
            top_k=0,
            stop="",
            max_new_tokens=1,
            caching_strategy=0,
            dump_csv=dump_csv,
        )

cache = llm1.process_batch(["My name is Alexander Hamilton and there's a million things I haven't done but "])
normal = llm2.process_batch(["My name is Alexander Hamilton and there's a million things I haven't done but "])

layer=0
tolerance = 1e-3

cache_kv = cache.past_key_values
normal_kv = normal.past_key_values

suf_len = cache.hidden_states[0].shape[1]
total_len = normal.hidden_states[0].shape[1]
pref_len = total_len - suf_len

print("suffix and total len: ", suf_len, " ", total_len)
print("number of hidden states:", len(normal.hidden_states))

cache_hs = cache.hidden_states[layer+1] # 0th layers hidden state is at index 1. At index 0, the initial hs is just input embeds. (Total 33 hs's for 32 layers)
normal_hs = normal.hidden_states[layer+1]

cache_q = cache.hidden_states[0]
normal_q = normal.hidden_states[0]

k_cache = cache_kv[layer][0]
k_normal = normal_kv[layer][0]

v_cache = cache_kv[layer][1]
v_normal = normal_kv[layer][1]

match_k = torch.allclose(k_cache, k_normal, atol=tolerance)
match_v = torch.allclose(v_cache, v_normal, atol=tolerance)
match_q = torch.allclose(cache_q, normal_q[:, pref_len:, :], atol=tolerance)
match_first_hs = torch.allclose(cache_hs, normal_hs[:, pref_len:, :], atol=tolerance)

print("k match? :", match_k)
print("v match? :", match_v)
print("q match? :", match_q)
print("match first hs? ", match_first_hs)