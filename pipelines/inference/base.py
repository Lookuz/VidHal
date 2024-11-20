import string
import re
import torch
from tqdm import tqdm

from ...dataset import VidHalDataset
from ...utils import generate_display_order

class VidHalInferencePipeline:
    """
    VidHalInferencePipeline and it's derivatives should handle:
    1. Formatting of prompt to be provided to the model.
    2. Generation of response from the given prompt and video.
    """
    def __init__(
        self, 
        model,
        dataset : VidHalDataset,
        option_display_order : dict = None,
        generation_config = {},
        *args, **kwargs
    ):
        self.model = model
        self.dataset = dataset
        self.generation_config = generation_config
        self.option_display_order = option_display_order if option_display_order is not None else generate_display_order(dataset)

    def reorder_options(self, captions, option_to_rank):
        """
        Re-orders the option prefixes (A, B, C) if there are less then the total number of captions presented to the model
        (e.g. When using only 2 of the M captions for relative caption ordering)
        """
        if len(captions) >= self.num_captions:
            return option_to_rank

        # Get options only if they exist in the captions
        option_to_rank = {option : rank for option, rank in option_to_rank.items() if rank in captions}
        # Adjust option letters
        option_prefix = list(string.ascii_uppercase)[:len(captions)]
        option_to_rank =  {option_prefix[i] : rank for i, (_, rank) in enumerate(
            sorted(list(option_to_rank.items()), key=lambda x: x[0])
        )}

        return option_to_rank

    def format_options_prompt(self, video_id, captions):
        """
        Generates the sub-prompt containing line-break separated [option : caption] to be displayed to the model
        """
        option_to_rank = self.option_display_order[video_id]
        option_to_rank = self.reorder_keys(captions, option_to_rank)
        options_prompt = "\n".join([f"{option}. {captions[rank]}" for option, rank in option_to_rank.items()])

        return options_prompt

    def format_prompt(
        self, 
        main_prompt, 
        options_prompt, 
        system_prompt=None, 
        *args, **kwargs):
        """
        NOTE: Implement this according to your model requirements
        """
        raise NotImplementedError
    
    def generate_response(
        self, 
        model, 
        video, 
        main_prompt, system_prompt=None,
        *args, **kwargs):
        """
        NOTE: Implement this according to your model requirements
        """
        raise NotImplementedError
    
    def run(self, save_path=None):
        raise NotImplementedError

class VidHalMCQAInferencePipeline(VidHalInferencePipeline):
    system_prompt_instruction = "You are provided with a video and a set of several captions. " \
        "Your task is to watch the video provided carefully, and select the caption that best describes the video. " \
        "Provide your answer only as a single letter representing the option whose caption that best describes the video, without any explanation."
    main_prompt_instruction = "Watch the video provided, and choose the option whose caption describes the video most accurately."
    def __init__(self, model, dataset, generation_config={}, *args, **kwargs):
        super().__init__(model, dataset, generation_config, *args, **kwargs)

    def process_response(self, response):
        """
        Parses the generated response to extract only the selected option.
        """
        last_option = list(string.ascii_uppercase)[self.num_captions - 1]
        match = re.search(fr"\b[a-{last_option.lower()}A-{last_option}]\b", response)
        match = match.group(0).upper().strip(";:., ") if match else None

        return match if match else response # If no match, keep original response in case model replies with caption instead of option
    
    def run(self, save_path=None):
        with torch.inference_mode(), torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                example = self.dataset[i]

class VidHalRelativeCOInferencePipeline(VidHalMCQAInferencePipeline):
    pass
