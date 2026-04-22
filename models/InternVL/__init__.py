import torch
from transformers import AutoModel, AutoTokenizer

from models.InternVL.processors.visual_processor import InternVL25VisualProcessor
from models.InternVL.processors.text_processor import InternVL25TextProcessor

def load_model(model_path=None, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_path = "OpenGVLab/InternVL2_5-8B" if model_path is None else model_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    return model, InternVL25VisualProcessor(), InternVL25TextProcessor(tokenizer=tokenizer)
