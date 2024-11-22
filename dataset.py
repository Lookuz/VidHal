import os 
import json

from torch.utils.data import Dataset
from utils import read_video

class VidHalDataset(Dataset):
    def __init__(self, data_path, video_root, vis_processor, num_frames, load_video=True) -> None:
        super().__init__()

        with open(data_path, "r") as f:
            self.examples = json.load(f)
        
        self.video_root = video_root
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.load_video = load_video
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        video_name, captions, aspect = example["video"], example["captions"], example["aspect"]
        video_path = os.path.join(self.video_root, f"{video_name}.mp4")

        if self.load_video:
            video, _, _ = read_video(video_path=video_path, num_frames=self.num_frames, sample="middle")
        else:
            video = None
        
        if video is not None and self.vis_processor is not None:
            video = self.vis_processor(video)

        return {
            "video" : video, "video_id" : video_name, "video_path" : video_path,
            "captions" : captions, "aspect" : aspect
        }
    