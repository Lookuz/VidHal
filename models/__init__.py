
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
    else:
        return {
            "random" : (None, None, None),
            # Proprietary models
            "gpt-4o" : lambda *x, **y : ("gpt-4o", None, None),
            "gemini-1.5-pro" : lambda *x, **y : ("gemini-1.5-pro", None, None),
            "gemini-1.5-flash" : lambda *x, **y : ("gemini-1.5-flash", None, None)
        }[model](*args, **kwargs)
