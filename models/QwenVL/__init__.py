import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_model(model_path=None, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_path = "OpenGVLab/Qwen-2.5-VL-7B" if model_path is None else model_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        device_map=device_map,
        torch_dtype="auto", 
        attn_implementation="flash_attention_2"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)

    return model, None, processor