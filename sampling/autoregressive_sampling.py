import torch

from tqdm import tqdm
from sampling.utils import norm_logits, sample

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = len(x)
    T = len(x) + N

    past_key_values = None
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
    return x

