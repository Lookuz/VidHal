import os
import base64
import cv2
from openai import OpenAI

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class GPT4oInferencePipeline(VidHalInferencePipeline):
    def __init__(self, model, api_key, dataset : VidHalDataset, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.client = OpenAI(api_key=api_key)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def encode_frames(self, video_path, seconds_per_frame=1):
        base64Frames = []

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame=0

        # Loop through the video and extract frames at specified sampling rate
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()

        return base64Frames
    
    def generate_response(self, main_prompt, system_prompt=None, image_path=None, *args, **kwargs):
        # Text only
        if image_path is None:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role" : "user", "content" : main_prompt}
                ]
            )

            return response.choices[0].message.content
        
        _, file_extension = os.path.splitext(image_path)

        # Video input to GPT as multiple frames
        if file_extension == ".mp4":
            frames = self.encode_frames(video_path=image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}'}}, frames),
                        {"type": "text", "text": main_prompt}
                    ]}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        
        # Single image input to GPT
        else:
            image = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model= self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image}"}
                        },
                        {"type": "text", "text": main_prompt},
                    ]}
                ],
                temperature=0.0,
            )

            return response.choices[0].message.content

class GPT4oMCQAInferencePipeline(GPT4oInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

class GPT4oNaiveOrderingInferencePipeline(GPT4oInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

class GPT4oRelativeOrderingInferencePipeline(GPT4oInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)
