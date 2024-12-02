import string
import re
import json
import torch
from tqdm import tqdm

from dataset import VidHalDataset
from utils import generate_display_order

class VidHalInferencePipeline:
    """
    VidHalInferencePipeline and it's derivatives should handle:
    1. Formatting of prompt to be provided to the model.
    2. Generation of response from the given prompt and video.
    """
    system_prompt_instruction = ""
    main_prompt_instruction = ""
    def __init__(
        self, 
        model,
        dataset : VidHalDataset,
        num_captions = 3,
        option_display_order : dict = None,
        generation_config = {},
        *args, **kwargs
    ):
        self.model = model
        self.dataset = dataset
        self.generation_config = generation_config
        self.num_captions = num_captions
        if option_display_order is None:
            print("No pre-defined option randomization supplied, generating one...")
            option_display_order = generate_display_order(dataset)
        self.option_display_order = option_display_order

    def format_options_prompt(self, captions, video_id=None, option_to_rank=None):
        """
        Generates the sub-prompt containing line-break separated [option : caption] to be displayed to the model
        """
        assert option_to_rank is not None or video_id is not None # Either video ID provided to use pre-defined ordering, or custom option ordering must be provided
        if option_to_rank is None:
            option_to_rank = self.option_display_order[video_id]
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

        Expected return type:
            prompts (tuple): Consisting of (main_prompt, system_prompt). If only one prompt is used, system prompt can be left optionally empty
        """
        raise NotImplementedError
    
    def generate_response(
        self, 
        video, 
        main_prompt, system_prompt=None,
        generation_config={},
        *args, **kwargs):
        """
        NOTE: Implement this according to your model requirements

        Expected return type:
            response (str) : Response generated by the model.
        """
        raise NotImplementedError

    def process_response(self, response):
        return response
    
    def run(self, save_path=None):
        responses = {}
        with torch.inference_mode(), torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                example = self.dataset[i]
                video, video_id, captions, video_path = example["video"], example["video_id"], example["captions"], example["video_path"]

                # Format caption options to be displayed to the model
                options_prompt = self.format_options_prompt(captions=captions, video_id=video_id)
                main_prompt, system_prompt = self.format_prompt(
                    self.main_prompt_instruction, options_prompt, self.system_prompt_instruction
                )

                # Generate response from the model
                response = self.generate_response(
                   video=video, main_prompt=main_prompt, system_prompt=system_prompt, 
                   generation_config=self.generation_config,
                   # For proprietary models
                   image_path=video_path
                )
                response = self.process_response(response)

                responses[video_id] = response

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(responses, f, indent=4)

class VidHalMCQAInferencePipeline(VidHalInferencePipeline):
    system_prompt_instruction = "You are provided with a video and a set of several captions. " \
        "Your task is to watch the video provided carefully, and select the caption that best describes the video. " \
        "Provide your answer only as a single letter representing the option whose caption that best describes the video, without any explanation."
    main_prompt_instruction = "Watch the video provided, and choose the option whose caption describes the video most accurately."

    def process_response(self, response):
        """
        Parses the generated response to extract only the selected option.
        """
        last_option = list(string.ascii_uppercase)[self.num_captions - 1]
        match = re.search(fr"\b[a-{last_option.lower()}A-{last_option}]\b", response)
        match = match.group(0).upper().strip(";:., ") if match else None

        return match if match else response # If no match, keep original response in case model replies with caption instead of option

class VidHalRelativeOrderingInferencePipeline(VidHalMCQAInferencePipeline):
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
    
    def prompt_paired_question(self, video, captions, options, option_to_rank):
        # Reorder keys (e.g. A, C -> A, B) and track the mapping
        display_options = list(string.ascii_uppercase)[:len(options)]
        remapped_option_to_rank = {display_options[i] : option_to_rank[option] for i, option in enumerate(options)}
        remapped_to_original = {display_options[i] : option for i, option in enumerate(options)}

        # Format prompt and generate response
        options_prompt = self.format_options_prompt(captions=captions, option_to_rank=remapped_option_to_rank)
        main_prompt, system_prompt = self.format_prompt(
            self.main_prompt_instruction, options_prompt, self.system_prompt_instruction
        )
        response = self.generate_response(
            self.model, video, main_prompt=main_prompt, system_prompt=system_prompt, generation_config=self.generation_config
        )

        # Process response and map back to original
        response = self.process_response(response)
        response = remapped_to_original[response]

        return response
    
    def prompt_relative_ordering(self, video, video_id, captions):
        overall_order = []
        # Transform from rank -> caption to option -> caption
        option_to_rank = self.option_display_order[video_id]
        options = sorted(list(option_to_rank.keys()))
        for option_A, option_B in zip(options, options[1:]):
            response = self.prompt_paired_question(video, captions, [option_A, option_B], option_to_rank)
            # Assign incorrect order if response is invalid or incorrect
            correct_order = [x[0] for x in sorted([
                (option_A, option_to_rank[option_A]), (option_B, option_to_rank[option_B])
            ], key=lambda x : int(x[-1]))]
            correct_answer =  correct_order[0]
            relative_order = correct_order if response == correct_answer else list(reversed(correct_order))

            if len(overall_order) < 1: 
                overall_order = relative_order
            elif overall_order[0] == relative_order[-1]: # Front prepend
                overall_order = relative_order[:1] + overall_order
            elif overall_order[-1] == relative_order[0]: # Back append
                overall_order = overall_order + relative_order[1:]
            # Intermediate insertion
            else:
                option_A, option_B = relative_order
                # Determine start point of insertion based on position of which key is present
                if option_A in overall_order:
                    index = overall_order.index(option_A) 
                    elements_to_compare = overall_order[index + 1:]
                else:
                    index = overall_order.index(option_B)  
                    elements_to_compare = list(reversed(overall_order[:index]))
                
                target_option = option_B if option_A in overall_order else option_A
                # Compare with candidates til unique ordering can be constructed
                for i, candidate_option in enumerate(elements_to_compare):
                    response = self.prompt_paired_question(video, captions, sorted([target_option, candidate_option]), option_to_rank)
                    if not response: # Select wrong answer if invalid one provided
                        response = sorted([
                            (target_option, option_to_rank[target_option]), (candidate_option, option_to_rank[candidate_option])
                        ], key=lambda x : -int(x[-1]))[0][0]

                    if (target_option == option_A and response != target_option) or (target_option == option_B and response == target_option):
                        new_subsequence = elements_to_compare[:i] + [target_option] + elements_to_compare[i:]
                        if target_option == option_B:
                            overall_order = overall_order[:index + 1] + new_subsequence
                        else:
                            overall_order = list(reversed(new_subsequence)) + overall_order[index:]
                        break
                
                # Insert at ends of list if not inserted
                if target_option not in overall_order:
                    overall_order = [target_option] + overall_order if target_option == option_A else overall_order + [target_option]

        return overall_order

    def run(self, save_path=None):
        responses = {}
        with torch.inference_mode(), torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                example = self.dataset[i]
                video, video_id, captions = example["video"], example["video_id"], example["captions"]
                predicted_order = self.prompt_relative_ordering(video, video_id, captions)
                responses[video_id] = predicted_order

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(responses, f, indent=4)

class VidHalNaiveOrderingInferencePipeline(VidHalInferencePipeline):
    system_prompt_instruction = "You are provided with a video and a set of several captions. " \
        "Your task is to order the captions in order of most to least relevant based on their alignment with the contents of the video. " \
        "Provide your answer without any further explanation."
    main_prompt_instruction = "Watch the video provided, and rank the captions below in order from the most accurate to the least accurate in describing the video. " \
        "Provide your response only as a sequence of comma separated option letters matching the corresponding captions. " \
        "Do not give any additional explanation for your answer."
    main_prompt_hint = "For example, if option B contains the caption that best describes the video, option A contains the caption that describes the video second best and " \
        "option C contains the caption that describes the video least accurately, provide your response as: B, A, C."
    def __init__(
        self, model, dataset: VidHalDataset, 
        num_captions=3, option_display_order: dict = None, 
        generation_config={},
        use_hint=True, # Add hint prompt into the main prompt
        *args, **kwargs):
        super().__init__(model, dataset, num_captions, option_display_order, generation_config, *args, **kwargs)

        self.use_hint = use_hint

    def format_prompt(
        self, main_prompt, options_prompt, system_prompt=None, *args, **kwargs
    ):
        if self.use_hint:
            main_prompt = f"{main_prompt}\n{self.main_prompt_hint}"
        return super().format_prompt(main_prompt, options_prompt, system_prompt, *args, **kwargs)

    def process_response(self, response):
        def condense_sequence(sequence):
            """
            Reduces consecutively repeating options, in cases where model explains option chosen
            """
            condensed_sequence = []
            
            for letter in sequence:
                if not condensed_sequence or condensed_sequence[-1] != letter:
                    condensed_sequence.append(letter)
            
            return condensed_sequence
        
        # Insert commas if letter sequence doesn't have
        response = re.sub(r'(?<=[A-Z])(?=[A-Z])', ', ', response)

        matches = re.findall(r"\b[A-Z]\b", response)
    
        # Convert matches to uppercase and remove duplicates while preserving order
        matches = [match.upper().strip(";:., ") for match in matches]
        matches = condense_sequence(matches) # Remove repeated consecutive letters due to explanations or descriptions
        # Handle more options than expected (e.g A, B, C, D, E, F, ...)
        valid_options = list(string.ascii_uppercase)[:self.num_captions]
        matches = [x for x in matches if x in valid_options]
        if len(matches) == self.num_captions:
            return matches
        else:
            initial_match = matches
        
        # Handle response with more constraints
        matches = re.findall(r"\b[A-Z][:\.,]", response)
        matches = condense_sequence([match.upper().strip(";:., ") for match in matches])
        matches = [x for x in matches if x in valid_options]
        if matches and len(matches) <= self.num_captions:
            return matches

        # Capture more than 3 letters - Response contains descriptory/explanatory elements
        if len(matches) > self.num_captions:
            # Break down by paragraph-level parsing
            sentences = response.split("\n")
            matches = [re.findall(r"(?<![a-zA-Z'])[A-Z]\b", x) for x in sentences]
            matches = [x for x in matches if len(x) > 1 and len(x) <= self.num_captions]

            # Break down into sentence-level parsing
            sentences = response.split(".")
            matches.extend([
                re.findall(r"(?<![a-zA-Z'])[A-Z]\b", x) for x in sentences if (
                    len(re.findall(r"(?<![a-zA-Z'])[A-Z]\b", x)) > 1  
                )
            ])
            matches = [[x for x in match if x in valid_options] for match in matches]

            # Condense duplicate orderings and get ordering with most
            matches = sorted(
                list(set([tuple(x) for x in matches])), key=lambda x: -len(x)
            )
            # Handle no valid ordering at the end
            try:
                matches = list(matches[0])
            except:
                matches = []

        if len(matches) <= self.num_captions and len(initial_match) > len(matches):
            return initial_match
        
        return matches
