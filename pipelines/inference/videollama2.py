import torch

from models.VideoLLaMA2.utils.mm_utils import tokenizer_multimodal_token, KeywordsStoppingCriteria
from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.VideoLLaMA2.processors.visual_processor import VideoLLaMA2VisualProcessor
from models.VideoLLaMA2.processors.text_processor import VideoLLaMA2TextProcessor

class VideoLLaMA2InferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, vis_processor : VideoLLaMA2VisualProcessor, text_processor : VideoLLaMA2TextProcessor,
        num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return self.text_processor.prepare_prompt(f"{main_prompt}\n\n{options_prompt}"), system_prompt
    
    def generate_response(
        self, video, main_prompt, system_prompt=None, 
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=128,
        num_return_sequences=1,
        num_beams=1,
        *args, **kwargs
    ):
        input_ids = tokenizer_multimodal_token(
            main_prompt, self.text_processor.tokenizer, self.text_processor.visual_token, return_tensors="pt"
        ).unsqueeze(0).long().cuda()
        attention_masks = input_ids.ne(self.text_processor.tokenizer.pad_token_id).long().cuda()

        stopping_criteria = KeywordsStoppingCriteria([
            self.text_processor.tokenizer.eos_token
        ], self.text_processor.tokenizer, input_ids)

        with torch.no_grad(), torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=[(video.half().cuda(), self.text_processor.modality)],
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                pad_token_id=self.text_processor.tokenizer.eos_token_id,
            )
        
        outputs = self.text_processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = outputs[0].strip() if len(outputs) <= 1 else [x.strip() for x in outputs]

        return outputs

class VideoLLaMA2MCQAInferencePipeline(VideoLLaMA2InferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class VideoLLaMA2NaiveOrderingInferencePipeline(VideoLLaMA2InferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class VideoLLaMA2RelativeOrderingInferencePipeline(VideoLLaMA2InferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
