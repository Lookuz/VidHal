from .model import load_pretrained_model
from models.VideoLLaMA2.utils.mm_utils import get_model_name_from_path
from models.VideoLLaMA2.processors.visual_processor import VideoLLaMA2VisualProcessor
from models.VideoLLaMA2.processors.text_processor import VideoLLaMA2TextProcessor

def load_model(model_path=None, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        load_4bit=load_4bit, load_8bit=load_8bit, device_map=device_map,
        **kwargs
    )

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    vis_processor = VideoLLaMA2VisualProcessor(processor=processor)
    text_processor = VideoLLaMA2TextProcessor(tokenizer=tokenizer, model_type=model.config.model_type)

    return model, vis_processor, text_processor
