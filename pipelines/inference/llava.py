import torch
from transformers import AutoConfig

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.LLaVA.processors.text_processor import LLaVANeXTTextProcessor, SeparatorStyle
from models.LLaVA.processors.visual_processor import LLaVANeXTVideoVisualProcessor
from models.LLaVA.utils.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from models.LLaVA.llavavid.constants import *

class LLaVANeXTVideoInferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, vis_processor : LLaVANeXTVideoVisualProcessor, text_processor : LLaVANeXTTextProcessor, model_path=None,
        num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        if model_path is not None:
            self.config = AutoConfig.from_pretrained(model_path)
        else:
            self.config = None

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, answer_prompt=None, *args, **kwargs):
        prompt = f"{main_prompt}\n\n{options_prompt}"
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conversation = self.text_processor.copy()
        conversation.system = system_prompt
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], answer_prompt)
        prompt = conversation.get_prompt()

        return prompt, None
    
    def generate_response(
        self, video, main_prompt, system_prompt=None, modalities="video", 
        do_sample=False, 
        temperature=0.0, 
        max_new_tokens=1024, 
        top_p=0.1, 
        num_beams=1, 
        use_cache=True, 
        num_return_sequences=1, 
        *args, **kwargs
    ):
        if len(video.shape) > 4:
            video = video.squeeze(0)
        video = [video]

        input_ids = tokenizer_image_token(main_prompt, self.text_processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        if self.text_processor.tokenizer.pad_token_id is None:
            if "qwen" in self.text_processor.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.text_processor.tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(self.text_processor.tokenizer.pad_token_id).long().cuda()

        stop_str = self.text_processor.sep if self.text_processor.sep_style != SeparatorStyle.TWO else self.text_processor.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.text_processor.tokenizer, input_ids)

        with torch.no_grad(), torch.inference_mode():
            if self.config is not None and "mistral" not in self.config._name_or_path.lower():
                output_ids = self.model.generate(
                    inputs=input_ids, images=video, attention_mask=attention_masks, 
                    modalities=modalities, do_sample=do_sample, 
                    temperature=temperature, max_new_tokens=max_new_tokens, 
                    top_p=top_p, num_beams=num_beams, use_cache=use_cache, 
                    stopping_criteria=[stopping_criteria],
                    num_return_sequences=num_return_sequences
                )
            else:
                output_ids = self.model.generate(
                    inputs=input_ids, images=video, attention_mask=attention_masks, 
                    modalities=modalities, do_sample=do_sample, 
                    temperature=temperature, max_new_tokens=max_new_tokens, 
                    top_p=top_p, num_beams=num_beams, use_cache=use_cache,
                    num_return_sequences=num_return_sequences
                )

        outputs = self.text_processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if self.config is not None and "mistral" not in self.config._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        
        return outputs

class LLaVANeXTVideoMCQAInferencePipeline(LLaVANeXTVideoInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, model_path=None, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, model_path, num_captions, option_display_order, generation_config, *args, **kwargs)

class LLaVANeXTVideoNaiveOrderingInferencePipeline(LLaVANeXTVideoInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, model_path=None, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, model_path, num_captions, option_display_order, generation_config, *args, **kwargs)

class LLaVANeXTVideoRelativeOrderingInferencePipeline(LLaVANeXTVideoInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, model_path=None, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, model_path, num_captions, option_display_order, generation_config, *args, **kwargs)
