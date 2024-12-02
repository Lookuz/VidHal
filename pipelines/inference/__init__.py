from pipelines.inference.base import VidHalInferencePipeline
from pipelines.inference.random import *
from pipelines.inference.videollama2 import *
from pipelines.inference.llava import *
from pipelines.inference.mplug_owl3 import *
from pipelines.inference.videochat2 import *
from pipelines.inference.gpt4 import *
from pipelines.inference.gemini import *

def get_inference_pipeline(name, task) -> VidHalInferencePipeline:
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
