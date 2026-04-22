import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class MiniCPPTextProcessor(nn.Module):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

def load_model(model_path=None, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_path = "openbmb/MiniCPM-V-2_6" if model_path is None else model_path
    model = AutoModel.from_pretrained(
        model_path, 
        device_map=device_map,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        attn_implementation='sdpa'
    ).eval()
    text_processor = MiniCPPTextProcessor(
        AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    )

    return model, None, text_processor
    