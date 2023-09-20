from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
from typing import Tuple
import torch.utils.benchmark as benchmark

from kvcache_model import KVCacheModel
from utils import norm_logits, sample
class Decoder:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def decode(self, t : torch.Tensor) -> str:
        # assert t.dim == 2, "t must be 2d tensor"
        return self.tokenizer.decode(t[0], skip_special_tokens=True)

DECODER : Decoder = None    


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for sample.py')

    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    parser.add_argument('--approx_model_name', type=str, default="/share_nfs/fangjiarui/root/code/hf_models/bloom-560m")
    parser.add_argument('--target_model_name', type=str, default="/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    args = parser.parse_args()
    return args



@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    with tqdm(total=N, desc="autoregressive sampling") as pbar:
        while n < T:
            # outputs = model(x)
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = torch.unsqueeze(last_ids, 0)
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                outputs = model(x)
            last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
            n += 1
            pbar.update(1)

    return x

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p), random_seed)
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]), random_seed)
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :], random_seed)
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    if verbose:
        global DECODER
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p, random_seed)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p, random_seed)
    
    torch.manual_seed(123)
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = prefix.shape[1]

            x = approx_model_cache.generate(prefix, gamma)
            _ = target_model_cache.generate(x, 1)
            
            n = prefix_len + gamma - 1
            for i in range(gamma):
                r = torch.rand(1, device = device)
                j = x[:, prefix_len + i]
                
                if r > (target_model_cache._prob_list[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_list[:, prefix_len + i - 1, j]):
                    # reject
                    n = prefix_len + i - 1
                    break
                
                if verbose:
                    print(f"approx guess accepted {j[0]}: \033[31m{DECODER.decode(torch.tensor([j]))}\033[0m")
            
            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            
            approx_model_cache.rollback(n+1)
            
            assert approx_model_cache._prob_list.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_list.shape}, n {n}"
            
            if n < prefix_len + gamma - 1:
                # reject someone, sample from the pos n
                t = sample(max_fn(target_model_cache._prob_list[:, n, :] - approx_model_cache._prob_list[:, n, :]), random_seed=random_seed)
                if verbose:
                    print(f"target resamples at position {n}: \033[34m{DECODER.decode(t)}\033[0m")
                
                target_model_cache.rollback(n+1)
            else:
                # all approx model decoding accepted
                assert n == target_model_cache._prob_list.shape[1] - 1
                t = sample(target_model_cache._prob_list[:, -1, :], random_seed=random_seed)
                if verbose:
                    print(f"target samples {n}: \033[35m{DECODER.decode(t)}\033[0m")
                target_model_cache.rollback(n+2)
            
            
            prefix = torch.cat((prefix, t), dim=1)
            
            if not verbose:
                pbar.update(n - pbar.n)

    return prefix

def generate(input_text, approx_model_name, target_model_name, num_tokens=40, verbose = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
    
    global DECODER
    DECODER = Decoder(tokenizer)
    
    print("begin loading models")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name).to(torch_device)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(torch_device)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 10
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"large (target) model autoregressive_sampling: {generated_text}")

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"small (approx) model autoregressive_sampling: {generated_text}")

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"google's speculative_sampling: {generated_text}")
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"deepmind's speculative_sampling: {generated_text}")

if __name__ == "__main__":
    args = parse_arguments()
    generate(args.input, args.approx_model_name, args.target_model_name, verbose=args.verbose)

    
