import re
import random
import numpy as np

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class RandomInferencePipeline(VidHalInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model=None, num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt
    
    def generate_response(self, video, main_prompt, system_prompt=None, generation_config=..., *args, **kwargs):
        if "choose" in main_prompt:
            options = list(set(re.findall(r'\b[A-Z]\b', main_prompt)))
            return random.choice(options)
        else:
            return ", ".join(np.random.permutation(["A", "B", "C"]).tolist()) 

class RandomMCQAInferencePipeline(RandomInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model=None, num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, num_captions, option_display_order, generation_config, *args, **kwargs)

class RandomNaiveOrderingInferencePipeline(RandomInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model=None, num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, num_captions, option_display_order, generation_config, *args, **kwargs)

class RandomRelativeOrderingInferencePipeline(RandomInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model=None, num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, num_captions, option_display_order, generation_config, *args, **kwargs)
