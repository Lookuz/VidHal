from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.VideoChat2.processors.text_processor import VideoChat2ChatProcessor
from models.VideoChat2.processors.visual_processor import VideoChat2VisualProcessor
from models.VideoChat2.utils.easydict import EasyDict

class VideoChat2InferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, 
        model, vis_processor : VideoChat2VisualProcessor, text_processor : VideoChat2ChatProcessor,
        num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt="", *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(
        self, video, main_prompt, system_prompt="", 
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=128,
        num_return_sequences=1,
        num_beams=1,
        *args, **kwargs
    ):
        conversation = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        # Construct text prompt
        conversation.messages.append([conversation.roles[0], f"<Video><VideoHere></Video>\n"])
        conversation = self.text_processor.ask(system_prompt + main_prompt, conversation)

        if len(video.shape) < 5:
            video = video.unsqueeze(0) # Add batch dimension
        video = video.to(self.model.device)

        if system_q:
            video_emb, _ = self.model.encode_visual_features(video, system_prompt + main_prompt)
        else:
            video_emb, _ = self.model.encode_visual_features(video, system_prompt)
        video_list = [video_emb]

        # Generate response
        response, _, _ = self.text_processor.answer(
            conv=conversation, model=self.model, video_embs=video_list, 
            answer_prompt=answer_prompt,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
        )
        if isinstance(response, list):
            response = return_prompt + "".join([x.strip() for x in response])
        else:
            response = return_prompt + response.strip().split('\n')[0]

        return response

class VideoChat2MCQAInferencePipeline(VideoChat2InferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class VideoChat2NaiveOrderingInferencePipeline(VideoChat2InferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class VideoChat2RelativeOrderingInferencePipeline(VideoChat2InferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
