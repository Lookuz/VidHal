from torch import nn

class InternVL25TextProcessor(nn.Module):
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer

    def forward(self, text):
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        return inputs