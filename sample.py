from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
from typing import Tuple

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

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch, 1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        raise RuntimeError
    return idx_next

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break
    
def trim_kv_cache(past_key_values : Tuple[Tuple[torch.Tensor, torch.Tensor]], q : torch.Tensor, end_pos : int):
    """
    trim the KV cache to the end_pos

    Args:
        past_key_values (Tuple): KV Cache
        end_pos (int): the position of the valid prefix

    Returns:
        Tuple: the trimmed KV Cache
    """
    past_key_values_trimmed = []
    for kv in past_key_values:
        k, v = kv
        # NOTE() the indexing is specific for bloom. This won't work for other models
        # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
        k = k[:, :, :end_pos]
        v = v[:, :end_pos, :]
        kv_trimmed = (k, v)
        past_key_values_trimmed.append(kv_trimmed)
    
    q = q[:, :end_pos, :]
    return past_key_values_trimmed, q
        
    
def forward_with_kvcache(model, input_ids, past_key_values, cached_q, temperature, top_k, top_p, use_debug = False):
    if past_key_values is None:
        assert cached_q is None
        # the first forward returns the prompt's logits
        outputs = model(input_ids)
        cached_q = outputs.logits
        for i in range(cached_q.shape[-2]):   
            cached_q[:, i, :] = norm_logits(cached_q[:, i, :], temperature, top_k, top_p)
        last_q = cached_q[:, -1, :]
    else:
        # return the last token's logits
        cached_len = 0
        for kv in past_key_values:
            k, v = kv
            cached_len = k.shape[2]
            
        last_input_id = input_ids[:, cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)
        
        if use_debug:
            print(f"last_input_id shape {last_input_id.shape}")
            _debug_show_kvcache(past_key_values)
        
        outputs = model(last_input_id, past_key_values=past_key_values, use_cache=True)
        
        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
        for i in range(not_cached_q.shape[-2]):   
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temperature, top_k, top_p)    
            
        if cached_q is not None:
            cached_q = torch.cat([cached_q, not_cached_q], dim=1)
        last_q = not_cached_q[:, -1, :]
    
    return last_q, outputs.past_key_values, cached_q

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    with tqdm(total=N, desc="autoregressive sampling") as pbar:
        while n < T:
            # outputs = model(x)
            last_q, past_key_values, _ = forward_with_kvcache(model, x, past_key_values, None, temperature, top_k, top_p)
            # logits = outputs.logits[::, -1, :]
            # past_key_values = outputs.past_key_values
            idx_next = sample(last_q)
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

    
def generate_with_kvcache(prefix : torch.Tensor, 
                                 gamma : int, 
                                 approx_model : torch.nn.Module,
                                 temperature : float, 
                                 top_k : float, 
                                 top_p : float,
                                 past_key_values : Tuple[Tuple[torch.Tensor, torch.Tensor]] = None,
                                 cached_q = None,
                                 use_debug = False) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor]]]:
    """ forward the model gamma times

    Args:
        prefix (torch.Tensor): the prefix
        gamma (int): how many times approx guesses
        approx_model (torch.nn.Module): an approx model
        temperature (float): temp for sampling
        top_k (float): top_k for sampling
        top_p (float): top_p for sampling
        past_key_values : valid kv cache
        cached_q: valid probability distribution of vocab on the all of the prefix tokens

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple]: prefix+generated tokens, past key value cache, probability distribution of vocab on the all of the tokens
    """
    x = prefix

    for _ in range(gamma):
        q, past_key_values, cached_q = forward_with_kvcache(approx_model, x, past_key_values, cached_q, temperature, top_k, top_p, use_debug)
        next_tok = sample(q)
        x = torch.cat((x, next_tok), dim=1)
    
    return x, past_key_values, cached_q

@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0) -> torch.Tensor:
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
                                  temperature, top_k, top_p))
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
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix

@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False) -> torch.Tensor:
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
    
    if verbose:
        global DECODER
    
    target_model_past_key_values = None
    approx_model_past_key_values = None
    #TODO() we can reduce the volume of q
    cached_q = None
    cached_p = None
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            prefix_len = prefix.shape[1]

            x, approx_model_past_key_values, cached_q = generate_with_kvcache(
                prefix, gamma, approx_model, 
                temperature, top_k, top_p, approx_model_past_key_values, cached_q)

            # q (batch_size, prefix_len+gamma, vocab)
            # assert q.shape[-2] == gamma, f"q.shape {q.shape} dose not match gamma {gamma}"
            
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            _, target_model_past_key_values, cached_p = forward_with_kvcache(
                                       target_model,
                                       x,
                                       target_model_past_key_values, cached_p, 
                                       temperature, top_k, top_p, 
                                       use_debug=False)
            
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            n = prefix_len + gamma - 1
            for i in range(gamma):
                r = torch.rand(1, device = cached_q.device)
                j = x[:, prefix_len + i]
                
                # print(f"cached_q {cached_q.shape}, p {cached_p.shape} prefix_len {prefix_len}")
                if r > (cached_p[:, prefix_len + i - 1, j]) / (cached_q[:, prefix_len + i - 1, j]):
                    # reject
                    n = prefix_len + i - 1
                    break
                
                if verbose:
                    print(f"approx guess accepted {j[0]}: \033[31m{DECODER.decode(torch.tensor([j]))}\033[0m")
            
            # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            approx_model_past_key_values, cached_q = trim_kv_cache(approx_model_past_key_values, cached_q, n+1)
            assert cached_q.shape[-2] <= n + 1, f"cached_q.shape {cached_q.shape}, n {n}"
            
            if n < prefix_len + gamma - 1:
                # reject someone, sample from the pos n
                t = sample(max_fn(cached_p[:, n, :] - cached_q[:, n, :]))
                if verbose:
                    print(f"target resamples {n}: \033[34m{DECODER.decode(t)}\033[0m")
                
                # target_model_past_key_values = None
                # cached_p = None
                target_model_past_key_values, cached_p = trim_kv_cache(target_model_past_key_values, cached_p, n+1)
            else:
                # all approx model decoding accepted
                assert n == cached_p.shape[1] - 1
                t = sample(cached_p[:, -1, :])
                if verbose:
                    print(f"target samples {n}: \033[35m{DECODER.decode(t)}\033[0m")
                target_model_past_key_values, cached_p = trim_kv_cache(target_model_past_key_values, cached_p, n+2)
            
            
            prefix = torch.cat((prefix, t), dim=1)
            
            if not verbose:
                pbar.update(n - pbar.n)

    return prefix

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, verbose = False):
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
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"deepmind's speculative_sampling: {generated_text}")


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


if __name__ == "__main__":
    args = parse_arguments()
    generate(args.input, args.approx_model_name, args.target_model_name, verbose=args.verbose)

    
