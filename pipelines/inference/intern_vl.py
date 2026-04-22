import torch

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.InternVL.processors.visual_processor import InternVL25VisualProcessor
from models.InternVL.processors.text_processor import InternVL25TextProcessor

class InternVL25InferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, vis_processor : InternVL25VisualProcessor, text_processor : InternVL25TextProcessor,
        num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        num_frames = kwargs.get('num_frames', 16) 
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
        main_prompt = video_prefix + main_prompt
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(self, video, main_prompt, system_prompt=None, generation_config=..., *args, **kwargs):
        if generation_config is None:
            generation_config = {"do_sample" : False, "max_new_tokens" : 128}
        pixel_values, num_patches_list = video

        response, history = self.model.chat(
            self.text_processor.tokenizer, 
            pixel_values.to(device=self.model.device, dtype=torch.bfloat16), 
            main_prompt, 
            generation_config,
            num_patches_list=num_patches_list, 
            history=None, 
            return_history=True)

        return response

class InternVL25MCQAInferencePipeline(InternVL25InferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: InternVL25VisualProcessor, text_processor : InternVL25TextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class InternVL25NaiveOrderingInferencePipeline(InternVL25InferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: InternVL25VisualProcessor, text_processor : InternVL25TextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class InternVL25RelativeOrderingInferencePipeline(InternVL25InferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor: InternVL25VisualProcessor, text_processor : InternVL25TextProcessor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
