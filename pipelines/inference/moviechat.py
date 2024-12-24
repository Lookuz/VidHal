import cv2
from PIL import Image

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)
from models.MovieChat.conversation.conversation_video import Chat

class MovieChatInferencePipeline(VidHalInferencePipeline):
    def __init__(self, 
        dataset: VidHalDataset, model, vis_processor, text_processor : Chat,
        num_captions=3, option_display_order: dict = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.vis_processor = text_processor.vis_processor
        self.text_processor = text_processor

    def format_prompt(self, main_prompt, options_prompt, system_prompt="", *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt
    
    def get_first_frame(self, video_path):
        # Get first frame for long-video encoding
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fps)
        ret, frame = cap.read()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.text_processor.image_vis_processor(image).unsqueeze(0).unsqueeze(2).half().to(self.model.device)

        return image

    def generate_response(
        self, video, image_path, main_prompt, 
        system_prompt=None, 
        middle_video=False, # True for breakpoint mode
        cur_min=0, cur_sec=0, # Change for breakpoint mode (intermediate starting point for video)
        fragment_video_path=None,
        max_new_tokens=300, 
        num_beams=1, 
        min_length=1, 
        top_p=0.9,
        repetition_penalty=1.0, 
        length_penalty=1, 
        temperature=1.0, 
        max_length=2000,
        *args, **kwargs
    ):
        cur_image = self.get_first_frame(video_path=image_path).to(self.model.device)
        cur_image = self.model.encode_image(cur_image)
        
        video_emb = self.text_processor.upload_video_without_audio(
            video_path=image_path, 
            fragment_video_path=fragment_video_path,
            cur_min=cur_min, cur_sec=cur_sec,
            cur_image = cur_image, 
            middle_video = middle_video
        )

        response = self.text_processor.answer(
            img_list=[video_emb],
            input_text=main_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            max_length=max_length
        )

        return response

class MovieChatMCQAInferencePipeline(MovieChatInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class MovieChatNaiveOrderingInferencePipeline(MovieChatInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)

class MovieChatRelativeOrderingInferencePipeline(MovieChatInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def  __init__(self, dataset, model, vis_processor, text_processor, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(dataset, model, vis_processor, text_processor, num_captions, option_display_order, generation_config, *args, **kwargs)
