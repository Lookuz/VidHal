import os
import math
from transformers import AutoConfig

from models.LLaVA.llavavid.mm_utils import get_model_name_from_path
from models.LLaVA.llavavid.model.builder import load_pretrained_model
from models.LLaVA.processors.text_processor import get_text_processor
from models.LLaVA.processors.visual_processor import LLaVANeXTVideoVisualProcessor

def load_model(
    model_path, model_base=None, mode="video", 
    load_8bit=False, load_4bit=False, device_map="auto", 
    # Additional model parameters to override
    mm_spatial_pool_mode="average",
    mm_newline_position="no_token",
    mm_pooling_position="after",
    mm_spatial_pool_stride=2,
    num_frames=32,
    *args, **kwargs
):    
    model_path = os.path.expanduser(model_path)
    cfg_pretrained = AutoConfig.from_pretrained(model_path)
    model_name = get_model_name_from_path(model_path)

    overwrite_config = {
        "mm_spatial_pool_mode" : mm_spatial_pool_mode,
        "mm_spatial_pool_stride" : mm_spatial_pool_stride,
        "mm_newline_position" : mm_newline_position,
        "mm_pooling_position" : mm_pooling_position
    }

    # Interpolation for frame number and RoPE scaling
    if "qwen" not in model_path.lower():
        if "224" in cfg_pretrained.mm_vision_tower:
            # Suppose the length of text tokens is around 1000, from bo's report
            least_token_number = num_frames * (16 // mm_spatial_pool_stride) ** 2 + 1000
        else:
            least_token_number = num_frames * (24 // mm_spatial_pool_stride) ** 2 + 1000

        scaling_factor = math.ceil(least_token_number / 4096)
        if scaling_factor >= 2:
            if "vicuna" in cfg_pretrained._name_or_path.lower():
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

    tokenizer, model, vis_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_4bit=load_4bit, load_8bit=load_8bit, overwrite_config=overwrite_config, device_map=device_map
    )

    text_processor = get_text_processor({
        "LLaVA-NeXT-Video-7B" : "vicuna_v1",
        "LLaVA-NeXT-Video-7B-DPO" : "vicuna_v1",
        "LLaVA-NeXT-Video-7B-Qwen2" : "qwen_1_5",
        "LLaVA-NeXT-Video-34B" : "mistral_direct",
        "LLaVA-NeXT-Video-34B-DPO" : "mistral_direct",
        "LLaVA-NeXT-Video-32B-Qwen" : "qwen_1_5",
        "LLaVA-NeXT-Video-72B-Qwen2" : "qwen_1_5"
    }[model_name])
    text_processor.tokenizer = tokenizer

    vis_processor = LLaVANeXTVideoVisualProcessor(vis_processor)

    return model, vis_processor, text_processor
