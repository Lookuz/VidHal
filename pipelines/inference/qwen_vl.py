import torch

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from qwen_vl_utils import process_vision_info

class Qwen25VLInferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, vis_processor, text_processor,
        num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(self, video, main_prompt, system_prompt=None, generation_config=None, *args, **kwargs):
        if generation_config is None:
            generation_config = {"do_sample" : False, "max_new_tokens" : 128}

        # Remove video and load video separately using qwen_vl_utils
        del video; torch.cuda.empty_cache()
        messages =  [{
            "role": "system", "content": system_prompt if system_prompt else ""
        }, {
            "role": "user",
            "content": [
                {"type": "video", "video" : kwargs.get("image_path")},
                {"type": "text", "text": main_prompt},
            ],
        }]
        text = self.text_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.text_processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, **generation_config)
        output_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response = self.text_processor.batch_decode(output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if isinstance(response, list):
            response = response[0]

        return response

class Qwen25VLMCQAInferencePipeline(Qwen25VLInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class Qwen25VLNaiveOrderingInferencePipeline(Qwen25VLInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class Qwen25VLRelativeOrderingInferencePipeline(Qwen25VLInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset: VidHalDataset, model, vis_processor, text_processor, num_captions=3, option_display_order: dict = None, generation_config=None, *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
