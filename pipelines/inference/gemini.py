import google.generativeai as genai

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
        return genai.upload_file(path=video_path)
    
    def generate_response(self, main_prompt, system_prompt=None, image_path=None, *args, **kwargs):
        video_file = self.upload_file(image_path)

        while video_file.state.name == "PROCESSING":
            video_file = genai.get_file(video_file.name)

        try:
            response = self.client.generate_content([
                system_prompt, video_file, main_prompt]
            )

            video_file.delete()

            return response.text
        except:
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
