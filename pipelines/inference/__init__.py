from .base import VidHalInferencePipeline
from .random import *

def get_inference_pipeline(name, task) -> VidHalInferencePipeline:
    return {
        "random" : {
            "mcqa" : RandomMCQAInferencePipeline,
            "naive_ordering" : RandomNaiveOrderingInferencePipeline,
            "relative_ordering" : RandomRelativeOrderingInferencePipeline
        }
    }[name][task]
