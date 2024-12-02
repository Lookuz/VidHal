import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer

from models.mPLUG_Owl3.model import mPLUGOwl3Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(
    model_path, attn_implementation='sdpa',
    load_8bit=False, load_4bit=True,
    *args, **kwargs
):
    from models.mPLUG_Owl3.processors.text_processor import mPLUGOwl3ChatProcessor

    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = False
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

    model = mPLUGOwl3Model.from_pretrained(
        model_path, attn_implementation=attn_implementation, torch_dtype=torch.half, device_map="auto", **kwargs
    )
    # model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
    text_processor = mPLUGOwl3ChatProcessor(
        tokenizer=tokenizer, processor=processor, device=model.device
    )

    # For mPLUG-Owl3, visual and text processor are integrated together
    return model, None, text_processor
