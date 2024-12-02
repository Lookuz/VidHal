
from models import (
    VideoLLaMA2, 
    LLaVA, 
    mPLUG_Owl3, 
    VideoChat2
)

def load_model(model, *args, **kwargs):
    """
    Returns a triple of (model, vis_processor, text_processor). If your model does not require any of these, you may return None
    """
    return {
        "random" : (None, None, None),
        "videollama2" : VideoLLaMA2.load_model,
        "llava-next-video" : LLaVA.load_model,
        "mplug_owl3" : mPLUG_Owl3.load_model,
        "videochat2" : VideoChat2.load_model,
        # Proprietary models
        "gpt-4o" : lambda *x, **y : ("gpt-4o", None, None),
        "gemini-1.5-pro" : lambda *x, **y : ("gemini-1.5-pro", None, None),
        "gemini-1.5-flash" : lambda *x, **y : ("gemini-1.5-flash", None, None)
    }[model](*args, **kwargs)
