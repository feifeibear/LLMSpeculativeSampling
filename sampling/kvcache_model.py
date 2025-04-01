import torch
from typing import Optional
from transformers.cache_utils import DynamicCache

from sampling.utils import norm_logits, sample
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 1, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True) -> torch.Tensor:
        if self._past_key_values is None:
            assert self._prob_history is None, f"{self._prob_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            if isinstance(self._past_key_values, DynamicCache):
                cached_len = self._past_key_values.get_seq_length()
            else:
                cached_len = 0
                for kv in self._past_key_values:
                    k, v = kv
                    cached_len = k.shape[2]  # For Bloom
                    if k.dim() == 3:  # Handle standard (batch, heads, seq_len, dim) format
                        cached_len = k.shape[2]
                    else:
                        cached_len = k.shape[-2]
                    break
            # return the last token's logits
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos: int):
        if isinstance(self._past_key_values, DynamicCache):
            # Truncate DynamicCache
            new_cache = DynamicCache()
            for layer_idx in range(len(self._past_key_values.key_cache)):
                k = self._past_key_values.key_cache[layer_idx][..., :end_pos, :]
                v = self._past_key_values.value_cache[layer_idx][..., :end_pos, :]
                new_cache.key_cache.append(k)
                new_cache.value_cache.append(v)
            self._past_key_values = new_cache
        else:
            # Original tuple-based handling
            past_key_values_trimmed = []
            for kv in self._past_key_values:
                k, v = kv
                if isinstance(self._model, BloomForCausalLM):
                    k = k[:, :, :end_pos]
                    v = v[:, :end_pos, :]
                else:
                    k = k[..., :end_pos, :]
                    v = v[..., :end_pos, :]
                past_key_values_trimmed.append((k, v))
            self._past_key_values = past_key_values_trimmed
        
        self._prob_history = self._prob_history[:, :end_pos, :]
