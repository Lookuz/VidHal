import torch

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.LongVU.longvu.conversation import SeparatorStyle
from models.LongVU.longvu.mm_datautils import tokenizer_image_token,  KeywordsStoppingCriteria
from models.LongVU.longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from models.LongVU.processors.visual_processor import LongVUVisualProcessor
from models.LongVU.processors.text_processor import LongVUTextProcessor

class LongVUInferencePipeline(VidHalInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: LongVUVisualProcessor, text_processor: LongVUTextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(self, video, main_prompt, system_prompt=None, generation_config=None, *args, **kwargs):
        video, image_sizes = video
        conv = self.text_processor(main_prompt)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, 
            self.text_processor.tokenizer, 
            IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.text_processor.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=128
            )

        response = self.text_processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return response

class LongVUMCQAInferencePipeline(LongVUInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: LongVUVisualProcessor, text_processor: LongVUTextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
    
class LongVUNaiveOrderingInferencePipeline(LongVUInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: LongVUVisualProcessor, text_processor: LongVUTextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
        
class LongVURelativeOrderingInferencePipeline(LongVUInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: LongVUVisualProcessor, text_processor: LongVUTextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
