from torch import nn
import torch
import numpy as np

from ..longvu.mm_datautils import process_images

class LongVUVisualProcessor(nn.Module):
    def __init__(self, image_processor, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.image_processor = image_processor
        self.model_config = model_config

    def __call__(self, video, *args, **kwargs):
        """
        Process the video frames for LongVU model.
        """
        if isinstance(video, torch.Tensor):
            video = np.stack([x.permute(1, 2, 0).numpy() for x in video])

        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self.image_processor, self.model_config)
        video = [frame.unsqueeze(0) for frame in video]

        return video, image_sizes
