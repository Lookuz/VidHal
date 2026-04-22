import os
import numpy as np
import base64
import cv2
from together import Together

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class TogetherAIInferencePipeline(VidHalInferencePipeline):
    def __init__(self, model, api_key, dataset : VidHalDataset, num_captions=3, option_display_order = None, generation_config = {}, *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.client = Together()
        self.model = model

    def encode_frames(self, video_path, max_frames=8):
        base64Frames = []

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            video.release()
            return base64Frames

        # Calculate evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)

        for frame_idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        return base64Frames

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt

    def generate_response(self, video, main_prompt, system_prompt=None, image_path=None, *args, **kwargs):
        frames = self.encode_frames(video_path=image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role" : "user", "content" : [
                *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}'}}, frames),
                {"type" : "text", "text": main_prompt}
            ]}
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content

class TogetherAIMCQAInferencePipeline(TogetherAIInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config = {}, *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)
    
class TogetherAINaiveOrderingInferencePipeline(TogetherAIInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config = {}, *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

class TogetherAIRelativeOrderingInferencePipeline(TogetherAIInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config = {}, *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)
