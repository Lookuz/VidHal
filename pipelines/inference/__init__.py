from pipelines.inference.base import VidHalInferencePipeline
from pipelines.inference.random import *
# from pipelines.inference.gpt4 import *
from pipelines.inference.gemini import *

def get_inference_pipeline(name, task) -> VidHalInferencePipeline:
    # Lazy loading of modules due to differing requirements
    if name == "videollama2":
        from pipelines.inference.videollama2 import (
            VideoLLaMA2MCQAInferencePipeline, VideoLLaMA2NaiveOrderingInferencePipeline, VideoLLaMA2RelativeOrderingInferencePipeline
        )
    elif name == "llava-next-video":
        from pipelines.inference.llava import (
            LLaVANeXTVideoMCQAInferencePipeline, LLaVANeXTVideoNaiveOrderingInferencePipeline, LLaVANeXTVideoRelativeOrderingInferencePipeline
        )
    elif name == "mplug_owl3":
        from pipelines.inference.mplug_owl3 import (
            mPLUGOwl3MCQAInferencePipeline, mPLUGOwl3NaiveOrderingInferencePipeline, mPLUGOwl3RelativeOrderingInferencePipeline
        )
    elif name == "videochat2":
        from pipelines.inference.videochat2 import (
            VideoChat2MCQAInferencePipeline, VideoChat2NaiveOrderingInferencePipeline, VideoChat2RelativeOrderingInferencePipeline
        )
    elif "moviechat" in name:
        pass # TODO
    
    return {
        "random" : {
            "mcqa" : RandomMCQAInferencePipeline,
            "naive_ordering" : RandomNaiveOrderingInferencePipeline,
            "relative_ordering" : RandomRelativeOrderingInferencePipeline
        },
        "videollama2" : {
            "mcqa" : VideoLLaMA2MCQAInferencePipeline,
            "naive_ordering" : VideoLLaMA2NaiveOrderingInferencePipeline,
            "relative_ordering" : VideoLLaMA2RelativeOrderingInferencePipeline
        },
        "llava-next-video" : {
            "mcqa" : LLaVANeXTVideoMCQAInferencePipeline,
            "naive_ordering" : LLaVANeXTVideoNaiveOrderingInferencePipeline,
            "relative_ordering" : LLaVANeXTVideoRelativeOrderingInferencePipeline
        },
        "mplug_owl3" : {
            "mcqa" : mPLUGOwl3MCQAInferencePipeline,
            "naive_ordering" : mPLUGOwl3NaiveOrderingInferencePipeline,
            "relative_ordering" : mPLUGOwl3RelativeOrderingInferencePipeline
        },
        "videochat2" : {
            "mcqa" : VideoChat2MCQAInferencePipeline,
            "naive_ordering" : VideoChat2NaiveOrderingInferencePipeline,
            "relative_ordering" : VideoChat2RelativeOrderingInferencePipeline
        },
        "gpt-4o" : {
            "mcqa" : GPT4oMCQAInferencePipeline,
            "naive_ordering" : GPT4oNaiveOrderingInferencePipeline,
            "relative_ordering" : GPT4oRelativeOrderingInferencePipeline
        },
        "gemini-1.5-flash" : {
            "mcqa" : GeminiMCQAInferencePipeline,
            "naive_ordering" : GeminiNaiveOrderingInferencePipeline,
            "relative_ordering" : GeminiRelativeOrderingInferencePipeline
        },
        "gemini-1.5-pro" : {
            "mcqa" : GeminiMCQAInferencePipeline,
            "naive_ordering" : GeminiNaiveOrderingInferencePipeline,
            "relative_ordering" : GeminiRelativeOrderingInferencePipeline
        },
    }[name][task]
