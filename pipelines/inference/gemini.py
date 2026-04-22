import os
import time
import google.api_core.exceptions
import google.generativeai as genai
from tqdm import tqdm

from dataset import VidHalDataset
from pipelines.inference.base import (
    VidHalInferencePipeline,
    VidHalMCQAInferencePipeline,
    VidHalNaiveOrderingInferencePipeline,
    VidHalRelativeOrderingInferencePipeline
)

class GeminiInferencePipeline(VidHalInferencePipeline):
    def __init__(self, model, api_key, dataset : VidHalDataset, num_captions=3, option_display_order = None, generation_config=..., *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model_name=model,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )

    def upload_file(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0].replace("_", "-")
        return genai.upload_file(path=video_path, name=video_name)

    def format_prompt(self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs):
        return f"{main_prompt}\n\n{options_prompt}", system_prompt
    
    def generate_response(self, video, main_prompt, system_prompt=None, image_path=None, *args, **kwargs):        
        video_name = os.path.splitext(os.path.basename(image_path))[0].replace("_", "-")
        try:
            video_file = genai.get_file(f"{video_name}")
            print(f"Using cached video file: {video_file.name}, state: {video_file.state.name}")
        except Exception as e:
            video_file = self.upload_file(image_path)

        max_retries, retry_count = kwargs.get('max_retries', 5), 0

        while retry_count < max_retries:
            try:
                # Check if video is still processing
                if hasattr(video_file, 'state') and video_file.state.name == 'PROCESSING':
                    time.sleep(2)
                    continue

                response = self.client.generate_content([
                    system_prompt, video_file, main_prompt]
                )

                return response.text
                
            except google.api_core.exceptions.ResourceExhausted as e:
                # Rate limit hit (429 error)
                print(f"Rate limit hit, waiting 1 minute... (attempt {retry_count + 1})")
                time.sleep(60)
                retry_count += 1
                
            except Exception as e:
                print(f"Error generating response: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    break
                time.sleep(10)  # Short wait for other errors

        print(f"Max retries ({max_retries}) exceeded")
        return ""

class GeminiMCQAInferencePipeline(GeminiInferencePipeline, VidHalMCQAInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

class GeminiNaiveOrderingInferencePipeline(GeminiInferencePipeline, VidHalNaiveOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

class GeminiRelativeOrderingInferencePipeline(GeminiInferencePipeline, VidHalRelativeOrderingInferencePipeline):
    def __init__(self, model, api_key, dataset, num_captions=3, option_display_order=None, generation_config=..., *args, **kwargs):
        super().__init__(model, api_key, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)
