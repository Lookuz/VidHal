from models.LongVU.processors.visual_processor import LongVUVisualProcessor
from models.LongVU.processors.text_processor import LongVUTextProcessor
from models.LongVU.longvu.builder import load_pretrained_model

def load_model(model_path=None, load_8bit=False, load_4bit=False, device_map="auto", **kwargs):
    model_path = "Vision-CAIR/LongVU_Qwen2_7B" if model_path is None else model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "cambrian_qwen",
        load_4bit=load_4bit, load_8bit=load_8bit, device_map=device_map,
        use_flash_attn=True
    )
    model = model.eval()

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    vis_processor = LongVUVisualProcessor(image_processor=image_processor, model_config=model.config)
    text_processor = LongVUTextProcessor(tokenizer=tokenizer)

    return model, vis_processor, text_processor
