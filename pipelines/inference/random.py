import re
import random
import numpy as np
from .base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class RandomInferencePipeline(VidHalInferencePipeline):
    def __init__(self, dataset, model=None, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, option_display_order, generation_config, *args, **kwargs)

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}"
    
    def generate_response(self, model, video, main_prompt, system_prompt=None, generation_config=..., *args, **kwargs):
        if "order" in main_prompt:
            return ", ".join(np.random.permutation(["A", "B", "C"]).tolist()) 
        else:
            options = re.findall(r'\b[A-Z]\b', main_prompt)
            return random.choice(options)

class RandomMCQAInferencePipeline(RandomInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model=None, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, option_display_order, generation_config, *args, **kwargs)

class RandomNaiveOrderingInferencePipeline(RandomInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model=None, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, option_display_order, generation_config, *args, **kwargs)

class RandomRelativeOrderingInferencePipeline(RandomInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset, model=None, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, option_display_order, generation_config, *args, **kwargs)
