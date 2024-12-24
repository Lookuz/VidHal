import torch
from omegaconf import OmegaConf

# Source code
from models.MovieChat.common.registry import registry
from models.MovieChat.models.moviechat import MovieChat
from models.MovieChat.models.moviechatplus import MovieChat as MovieChatPlus
from models.MovieChat.conversation.conversation_video import Chat
from models.MovieChat.processors.video_processor import AlproVideoEvalProcessor

# Installed version
# from MovieChat.processors.video_processor import AlproVideoEvalProcessor
# from MovieChat.models.chat_model import Chat
# from MovieChat.models.moviechat import MovieChat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(
    config_path, *args, **kwargs
):
    config = OmegaConf.load(config_path)

    model_class = {
        "moviechat" : MovieChat,
        "moviechat+" : MovieChatPlus
    }[config.model.arch]

    model = model_class.from_config(config.model).to(device)

    vis_processor_cfg = config.datasets.webvid.vis_processor.train
    vis_processor = AlproVideoEvalProcessor(
        image_size=vis_processor_cfg.image_size,
        n_frms=vis_processor_cfg.n_frms
    )

    text_processor = Chat(model=model, vis_processor=vis_processor, device=device)

    # For MovieChat, embed visual processor into text processor
    return model, None, text_processor
