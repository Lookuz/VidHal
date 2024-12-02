import torch

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.mPLUG_Owl3.processors.text_processor import mPLUGOwl3ChatProcessor

class mPLUGOwl3InferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, text_processor : mPLUGOwl3ChatProcessor,
        num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt
    
    def generate_response(
        self, video, main_prompt, system_prompt=None, 
        max_new_tokens=128,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        num_return_sequences=1,
        num_beams=1,
        *args, **kwargs
    ):
        inputs = self.text_processor.process_inputs(video, main_prompt)

        inputs = inputs.to(self.model.device)
        inputs.update({
            'tokenizer': self.text_processor.tokenizer,
            'decode_text': True,
        })

        with torch.no_grad(), torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams
            )
            
        outputs = outputs[0].strip() if len(outputs) <= 1 else [x.strip() for x in outputs]

        return outputs


class mPLUGOwl3MCQAInferencePipeline(mPLUGOwl3InferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class mPLUGOwl3NaiveOrderingInferencePipeline(mPLUGOwl3InferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class mPLUGOwl3RelativeOrderingInferencePipeline(mPLUGOwl3InferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset, model, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
