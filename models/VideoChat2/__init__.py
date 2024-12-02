import logging
import torch
import numpy as np

from peft import get_peft_model, LoraConfig, TaskType
from models.VideoChat2.utils.config import Config
from models.VideoChat2.model.videochat import VideoChat2Model
from models.VideoChat2.model.videochat_mistral import VideoChat2Mistral
from models.VideoChat2.model.videochat_phi import VideoChat2Phi3

from models.VideoChat2.processors.visual_processor import VideoChat2VisualProcessor
from models.VideoChat2.processors.text_processor import VideoChat2ChatProcessor

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, num_frames=8, ckpt_num_frames=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # NOTE: Added to force checkpoint and current number of frames to be same (1) if image used as input
    if num_frames == 1: 
        ckpt_num_frames = 1

    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    if n_position != pre_n_position:
        T = ckpt_num_frames # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // num_frames) ** 0.5) # testing size
        if new_P != 14:
            logger.info(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            logger.info(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if num_frames != ckpt_num_frames:
        logger.info(f'Pretraining uses 4 frames, but current frame is {num_frames}')
        logger.info(f'Interpolate the position embedding')
        T = ckpt_num_frames # checkpoint frame
        new_T = num_frames # testing frame
        # interpolate
        P = int((n_position // num_frames) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def load_model(config_path, model_path=None, num_frames=16, resolution=224, device=device, test=False, *args, **kwargs):
    config = Config.from_file(config_path)
    config.model.vision_encoder.num_frames = 4
    # config.model.vision_encoder.num_frames = num_frames

    if "inputs" not in config:
        config["inputs"] = {
            "video_input" : {
                "random_aug" : not test
            },
            "image_res" : resolution
        }

    if "mistral" in config_path:
        model_cls = "mistral"
        model = VideoChat2Mistral(config=config.model).to(config.device)
    elif "phi" in config_path:
        model_cls = "phi"
        model = VideoChat2Phi3(config=config.model).to(config.device)
    else:
        model_cls = "llama"
        model = VideoChat2Model(config=config.model).to(config.device)

    model.vision_encoder.encoder.pos_embed = get_sinusoid_encoding_table(
        n_position=(resolution // 16) ** 2 * num_frames, num_frames=num_frames
    )

    target_modules = None if isinstance(model, VideoChat2Model) else [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=test, r=16, lora_alpha=32, lora_dropout=0., target_modules=target_modules
    )
    model.language_model = get_peft_model(model.language_model, peft_config)

    if model_path is not None:
        state_dict = torch.load(model_path)
    
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)

        logging.info(msg)

    vis_processor = VideoChat2VisualProcessor(config, test=test)
    text_processor = VideoChat2ChatProcessor(model=model_cls, device=device)

    return model, vis_processor, text_processor
