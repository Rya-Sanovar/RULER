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
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs
        )
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print(generated_texts)
        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        print(f"CACHE STRAT = {self.cache_strategy}, RESULT: {results}")
        return results
    
llm_cache = HuggingFaceModel(
            name_or_path="microsoft/Phi-3.5-mini-instruct",
            do_sample=False,
            repetition_penalty=1,
            temperature=1.0,
            top_p=1,
            # top_k=0,
            stop="",
            max_new_tokens=1,
            caching_strategy=1,
            dump_csv=dump_csv,
        )

llm_normal = HuggingFaceModel(
            name_or_path="microsoft/Phi-3.5-mini-instruct",
            do_sample=False,
            repetition_penalty=1,
            temperature=1.0,
            top_p=1,
            # top_k=0,
            stop="",
            max_new_tokens=1,
            caching_strategy=0,
            dump_csv=dump_csv,
        )

llm_cache.process_batch(["My name is Alexander Hamilton and there's a million things I haven't done but just"])
llm_normal.process_batch(["My name is Alexander Hamilton and there's a million things I haven't done but just"])
