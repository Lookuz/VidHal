from torch import nn

from ..longvu.conversation import conv_templates
from ..longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from ..longvu.mm_datautils import tokenizer_image_token

class LongVUTextProcessor(nn.Module):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

    def __call__(self, prompt):
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        return conv