import torch
from PIL import Image
from torchvision.transforms import ToPILImage, Resize

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class MiniCPMInferencePipeline(VidHalInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.to_pil_image = ToPILImage()
        self.resize = Resize((448, 448))  # Add resizing transform

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(self, video, main_prompt, system_prompt=None, generation_config=None, *args, **kwargs):
        # Resize and transform to list of PIL Images
        video = [self.to_pil_image(self.resize(frame)) for frame in video]
        messages = [{"role": "user", "content": video + [main_prompt]},]

        response = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.text_processor.tokenizer,
            use_image_id=False,
            max_slice_num=2
        )

        return response

class MiniCPMMCQAInferencePipeline(MiniCPMInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
    
class MiniCPMNaiveOrderingInferencePipeline(MiniCPMInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class MiniCPMRelativeOrderingInferencePipeline(MiniCPMInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
