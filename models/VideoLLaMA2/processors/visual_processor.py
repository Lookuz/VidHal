from torch import nn

class VideoLLaMA2VisualProcessor(nn.Module):
    def __init__(self, processor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.processor = processor

    def forward(self, x):
        return self.processor.preprocess(
            x, return_tensors="pt"
        )["pixel_values"].half().cuda()
