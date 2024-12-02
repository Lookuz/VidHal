from models.VideoLLaMA2.utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN

class VideoLLaMA2TextProcessor:
    def __init__(
        self, tokenizer, model_type="videollama2", modality="video"
    ) -> None:
        
        self.tokenizer = tokenizer
        self.modality = modality
        if modality == 'image':
            self.visual_token = DEFAULT_IMAGE_TOKEN
        elif modality == 'video':
            self.visual_token = DEFAULT_VIDEO_TOKEN

        self.system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ] if model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral'] else []

    def prepare_prompt(self, prompt):
        message = [{'role': 'user', 'content': self.visual_token + '\n' + prompt}]
        message = self.system_message + message

        full_prompt = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        return full_prompt
