
def load_model(model, *args, **kwargs):
    """
    Returns a triple of (model, vis_processor, text_processor). If your model does not require any of these, you may return None
    """
    # Lazy load models, due to different requirements
    if model == "videollama2":
        from models import VideoLLaMA2
        return VideoLLaMA2.load_model(*args, **kwargs)
    elif model == "llava-next-video":
        from models import LLaVA
        return LLaVA.load_model(*args, **kwargs) 
    elif model == "mplug_owl3":
        from models import mPLUG_Owl3
        return mPLUG_Owl3.load_model(*args, **kwargs) 
    elif model == "videochat2":
        from models import VideoChat2
        return VideoChat2.load_model(*args, **kwargs) 
    elif "moviechat" in model:
        from models import MovieChat
        return MovieChat.load_model(*args, **kwargs) 
    elif model == "intern-vl25":
        from models import InternVL
        return InternVL.load_model(*args, **kwargs)
    elif model == "qwen-vl25":
        from models import QwenVL
        return QwenVL.load_model(*args, **kwargs) 
    elif model == "minicpm":
        from models import MiniCPM
        return MiniCPM.load_model(*args, **kwargs)
    elif model == "longvu":
        from models import LongVU
        return LongVU.load_model(*args, **kwargs)
    elif "together" in model:
        return ({
            "together-Qwen2.5-VL-72B-Instruct" : "Qwen/Qwen2.5-VL-72B-Instruct",
        }[model], None, None)
    else:
        return {
            "random" : (None, None, None),
            # Proprietary models
            "gpt-4o" : lambda *x, **y : ("gpt-4o", None, None),
            "gpt-4.1" : lambda *x, **y : ("gpt-4.1", None, None),
            "gemini-1.5-pro" : lambda *x, **y : ("gemini-1.5-pro", None, None),
            "gemini-1.5-flash" : lambda *x, **y : ("gemini-1.5-flash", None, None),
            "gemini-2.5-pro" : lambda *x, **y : ("gemini-2.5-pro", None, None),
            "gemini-2.5-flash" : lambda *x, **y : ("gemini-2.5-flash", None, None)
        }[model](*args, **kwargs)
