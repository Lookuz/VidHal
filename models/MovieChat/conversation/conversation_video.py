"""
Conversation prompt template of Video-LLaMA.
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py 
"""
import os
import torch
import numpy as np
import decord
import subprocess
from decord import VideoReader
from moviepy.editor import *
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Any
import os
from models.MovieChat.common.registry import registry
from models.MovieChat.processors import Blip2ImageEvalProcessor
            
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

class Chat:
    def __init__(self, model, vis_processor, device):
        self.device = device
        self.n_samples = 128
        self.short_memory_length = 18
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def capture_video(self, video_path, per_video_length, n_stage):
        start_time = n_stage * per_video_length
        end_time = (n_stage+1) * per_video_length
        video = CompositeVideoClip([VideoFileClip(video_path).subclip(start_time,end_time)])
        return video
    
    def parse_video_fragment(self, video_path, video_length, n_stage = 0):
        decord.bridge.set_bridge("torch")
        per_video_length = video_length / self.n_samples
        fragment_video = self.capture_video(video_path, per_video_length, n_stage)
        return fragment_video
    
    def video_duration(self, filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)
    
    def load_video(self, video_path, n_frms=4, height=-1, width=-1):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=height, width=width)
        vlen = len(vr)
        start, end = 0, vlen

        n_frms = min(n_frms, vlen)
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()

        # get_batch -> T, H, W, C
        temp_frms = vr.get_batch(indices)
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
        return frms, msg

    def get_context_emb(self, input_text, img_list):
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers.###Human: <Video><ImageHere></Video>"
        prompt_2 = input_text
        prompt_3 = "###Assistant:"
        prompt = prompt_1 + " " + prompt_2 + prompt_3

        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    
    def answer(
        self, img_list, input_text, 
        max_new_tokens=300, 
        num_beams=1, 
        min_length=1, 
        top_p=0.9,
        repetition_penalty=1.0, 
        length_penalty=1, 
        temperature=1.0, 
        max_length=2000
    ):
        embs = self.get_context_emb(input_text, img_list) 

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p, 
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty, 
            temperature=temperature, 
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        return output_text, output_token.cpu().numpy()
    
    def cal_frame(self, video_length, cur_min, cur_sec, middle_video):
        per_frag_second = video_length / self.n_samples
        if middle_video:
            cur_seconds = cur_min * 60 + cur_sec
            num_frames = int(cur_seconds / per_frag_second)
            per_frame_second = per_frag_second / self.short_memory_length
            cur_frame = int((cur_seconds-per_frag_second*num_frames)/per_frame_second)
            return num_frames, cur_frame
        else:
            cur_frame = 0
            num_frames = int(video_length / per_frag_second)
            return num_frames, cur_frame

    def upload_video_without_audio(
        self, video_path, fragment_video_path, cur_image, middle_video, 
        cur_min=0, cur_sec=0
    ):
        video_length = self.video_duration(video_path) 
        num_frames, cur_frame = self.cal_frame(video_length, cur_min, cur_sec, middle_video)
        
        if num_frames == 0:
            video_fragment = self.parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=0, n_samples= self.n_samples)
            video_fragment, _ = self.load_video(
                video_path=fragment_video_path,
                n_frms=self.vis_processor.n_frms, 
                height=224,
                width=224,
            ) 
            video_fragment = self.vis_processor.transform(video_fragment)
            video_fragment = video_fragment.unsqueeze(0).to(self.device)
            self.model.encode_short_memory_frame(video_fragment, cur_frame)
        else:
            for i in range(num_frames): 
                video_fragment = self.parse_video_fragment(video_path=video_path, video_length=video_length, n_stage=i)
                
                video_fragment, _ = self.load_video(
                    video_path=fragment_video_path,
                    n_frms=self.vis_processor.n_frms, 
                    height=224,
                    width=224
                )
                video_fragment = self.vis_processor.transform(video_fragment) 
                video_fragment = video_fragment.unsqueeze(0).to(self.device)

                if middle_video and (i + 1)==num_frames:
                    self.model.encode_short_memory_frame(video_fragment, cur_frame)
                else:
                    self.model.encode_short_memory_frame(video_fragment)

        video_emb, _ = self.model.encode_long_video(cur_image, middle_video)

        return video_emb  
