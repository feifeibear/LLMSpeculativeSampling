import torch

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Decoder(metaclass=Singleton):
    def __init__(self):
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, s: str, return_tensors='pt') -> torch.Tensor:
        return self.tokenizer.encode(s, return_tensors=return_tensors)
    
    def decode(self, t: torch.Tensor) -> str:
        return self.tokenizer.decode(t[0], skip_special_tokens=True)