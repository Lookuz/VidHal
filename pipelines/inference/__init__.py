from pipelines.inference.base import VidHalInferencePipeline
from pipelines.inference.random import *
from pipelines.inference.gpt4 import *
from pipelines.inference.gemini import *

def get_inference_pipeline(name, task) -> VidHalInferencePipeline:
    # Lazy loading of modules due to differing requirements
    if name == "videollama2":
        from pipelines.inference.videollama2 import (
            VideoLLaMA2MCQAInferencePipeline, VideoLLaMA2NaiveOrderingInferencePipeline, VideoLLaMA2RelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : VideoLLaMA2MCQAInferencePipeline,
            "naive_ordering" : VideoLLaMA2NaiveOrderingInferencePipeline,
            "relative_ordering" : VideoLLaMA2RelativeOrderingInferencePipeline
        }[task]
    elif name == "llava-next-video":
        from pipelines.inference.llava import (
            LLaVANeXTVideoMCQAInferencePipeline, LLaVANeXTVideoNaiveOrderingInferencePipeline, LLaVANeXTVideoRelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : LLaVANeXTVideoMCQAInferencePipeline,
            "naive_ordering" : LLaVANeXTVideoNaiveOrderingInferencePipeline,
            "relative_ordering" : LLaVANeXTVideoRelativeOrderingInferencePipeline
        }[task]
    elif name == "mplug_owl3":
        from pipelines.inference.mplug_owl3 import (
            mPLUGOwl3MCQAInferencePipeline, mPLUGOwl3NaiveOrderingInferencePipeline, mPLUGOwl3RelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : mPLUGOwl3MCQAInferencePipeline,
            "naive_ordering" : mPLUGOwl3NaiveOrderingInferencePipeline,
            "relative_ordering" : mPLUGOwl3RelativeOrderingInferencePipeline
        }[task]
    elif name == "videochat2":
        from pipelines.inference.videochat2 import (
            VideoChat2MCQAInferencePipeline, VideoChat2NaiveOrderingInferencePipeline, VideoChat2RelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : VideoChat2MCQAInferencePipeline,
            "naive_ordering" : VideoChat2NaiveOrderingInferencePipeline,
            "relative_ordering" : VideoChat2RelativeOrderingInferencePipeline
        }[task]
    elif "moviechat" in name:
        from pipelines.inference.moviechat import (
            MovieChatMCQAInferencePipeline, MovieChatNaiveOrderingInferencePipeline, MovieChatRelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : MovieChatMCQAInferencePipeline,
            "naive_ordering" : MovieChatNaiveOrderingInferencePipeline,
            "relative_ordering" : MovieChatRelativeOrderingInferencePipeline
        }[task]
    elif "intern-vl25" in name:
        from pipelines.inference.intern_vl import (
            InternVL25MCQAInferencePipeline, InternVL25NaiveOrderingInferencePipeline, InternVL25RelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : InternVL25MCQAInferencePipeline,
            "naive_ordering" : InternVL25NaiveOrderingInferencePipeline,
            "relative_ordering" : InternVL25RelativeOrderingInferencePipeline
        }[task]
    elif "qwen-vl" in name:
        from pipelines.inference.qwen_vl import (
            Qwen25VLMCQAInferencePipeline, Qwen25VLNaiveOrderingInferencePipeline, Qwen25VLRelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : Qwen25VLMCQAInferencePipeline,
            "naive_ordering" : Qwen25VLNaiveOrderingInferencePipeline,
            "relative_ordering" : Qwen25VLRelativeOrderingInferencePipeline
        }[task]
    elif "minicpm" in name:
        from pipelines.inference.minicpm import (
            MiniCPMMCQAInferencePipeline, MiniCPMNaiveOrderingInferencePipeline, MiniCPMRelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : MiniCPMMCQAInferencePipeline,
            "naive_ordering" : MiniCPMNaiveOrderingInferencePipeline,
            "relative_ordering" : MiniCPMRelativeOrderingInferencePipeline
        }[task]
    elif "longvu" in name:
        from pipelines.inference.longvu import (
            LongVUMCQAInferencePipeline, LongVUNaiveOrderingInferencePipeline, LongVURelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : LongVUMCQAInferencePipeline,
            "naive_ordering" : LongVUNaiveOrderingInferencePipeline,
            "relative_ordering" : LongVURelativeOrderingInferencePipeline
        }[task]

    elif "together" in name:
        from pipelines.inference.together import (
            TogetherAIMCQAInferencePipeline, TogetherAINaiveOrderingInferencePipeline, TogetherAIRelativeOrderingInferencePipeline
        )
        return {
            "mcqa" : TogetherAIMCQAInferencePipeline,
            "naive_ordering" : TogetherAINaiveOrderingInferencePipeline,
            "relative_ordering" : TogetherAIRelativeOrderingInferencePipeline
        }[task]

    return {
        "random" : {
            "mcqa" : RandomMCQAInferencePipeline,
            "naive_ordering" : RandomNaiveOrderingInferencePipeline,
            "relative_ordering" : RandomRelativeOrderingInferencePipeline
        },
        "gpt-4o" : {
            "mcqa" : GPT4oMCQAInferencePipeline,
            "naive_ordering" : GPT4oNaiveOrderingInferencePipeline,
            "relative_ordering" : GPT4oRelativeOrderingInferencePipeline
        },
        "gpt-4.1" : {
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
        "gemini-2.5-flash" : {
            "mcqa" : GeminiMCQAInferencePipeline,
            "naive_ordering" : GeminiNaiveOrderingInferencePipeline,
            "relative_ordering" : GeminiRelativeOrderingInferencePipeline
        },
        "gemini-2.5-pro" : {
            "mcqa" : GeminiMCQAInferencePipeline,
            "naive_ordering" : GeminiNaiveOrderingInferencePipeline,
            "relative_ordering" : GeminiRelativeOrderingInferencePipeline
        }
    }[name][task]
