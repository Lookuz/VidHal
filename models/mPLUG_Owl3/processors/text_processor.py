import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mPLUGOwl3ChatProcessor:
    def __init__(
        self, tokenizer, processor, device=device
    ) -> None:
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    def process_inputs(
        self, images, # Images or videos
        prompt, system=None
    ):
        # Create conversation template
        messages = [{"role": "user", "content": f"""{system}."""}] if system is not None else []

        # Process inputs for model
        if len(images.shape) < 4 or (
            len(images.shape) == 4 and images.shape[0] == 1
        ): # Image case
            messages.extend([
                {"role": "user", "content": f"""<|image|>{prompt}."""},
                {"role": "assistant", "content": ""}
            ])
            images = [images] if len(images.shape) < 4 else images
            inputs = self.processor(messages, images=[images], videos=None)
        else:
            messages.extend([
                {"role": "user", "content": f"""<|video|>{prompt}."""},
                {"role": "assistant", "content": ""}
            ])
            images = [images] if len(images.shape) < 5 else images
            inputs = self.processor(messages, images=None, videos=images)

        return inputs
