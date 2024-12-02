from enum import auto, Enum
from typing import List, Any, Union
import re
from PIL import Image

from transformers import AutoTokenizer

"""
Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/conversation.py
"""
class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()

class LLaVANeXTTextProcessor:
    def __init__(self, 
        system : str, roles : List[str], messages : List[List[str]], offset : int,
        sep_style : SeparatorStyle.SINGLE, sep: str = "###", sep2: str = None, version: str = "Unknown",
        tokenizer_id: str = "", tokenizer: Any = None, 
        stop_str: Union[str, List[str]] = None, # Stop criteria (the default one is EOS token)
        stop_token_ids: List[int] = None,  # Stops generation if meeting any token in this list
        *args, **kwargs
    ):
        self.system = system
        self.roles = roles
        self.messages = messages
        self.offset = offset
        
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2
        self.version = version

        self.tokenizer_id = tokenizer_id
        self.tokenizer = tokenizer
        self.stop_str = stop_str
        self.stop_token_ids = stop_token_ids


    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})

            return self.tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], "Conversation should alternate user/assistant/user/assistant/..."
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, image_format="PNG"):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        
        return image
    
    def get_images(self, return_path=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path:
                            img = self.process_image(img, image_process_mode)
                        else:
                            images.append(img)
        return images

    def copy(self):
        return LLaVANeXTTextProcessor(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

def get_text_processor(model_name):
    return {
        "default": LLaVANeXTTextProcessor(
            system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        ),
        "v0": LLaVANeXTTextProcessor(
            system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        ),
        "v1": LLaVANeXTTextProcessor(
            system="A chat between a curious user and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        ),
        "vicuna_v1": LLaVANeXTTextProcessor(
            system="A chat between a curious user and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        ),
        "llama_2": LLaVANeXTTextProcessor(
            system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
            roles=("USER", "ASSISTANT"),
            version="llama_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="<s>",
            sep2="</s>",
        ),
        "mistral_instruct": LLaVANeXTTextProcessor(
            system="",
            roles=("USER", "ASSISTANT"),
            version="llama_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="",
            sep2="</s>",
        ),
        "mistral_orca": LLaVANeXTTextProcessor(
            system="""<|im_start|>system
        You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!""",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        ),
        "mistral_zephyr": LLaVANeXTTextProcessor(
            system="""<|system|>
        You are a helpful AI assistant.""",
            roles=("<|user|>\n", "<|assistant|>\n"),
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="</s>",
        ),
        "mistral_direct": LLaVANeXTTextProcessor(
            system="""<|im_start|>system
        Answer the questions.""",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        ),
        "plain": LLaVANeXTTextProcessor(
            system="",
            roles=("", ""),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.PLAIN,
            sep="\n",
        ),
        "v0_plain": LLaVANeXTTextProcessor(
            system="",
            roles=("", ""),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.PLAIN,
            sep="\n",
        ),
        "chatml_direct": LLaVANeXTTextProcessor(
            system="""<|im_start|>system
        Answer the questions.""",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        ),
        "llava_v0": LLaVANeXTTextProcessor(
            system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
        ),
        "llava_v0_mmtag": LLaVANeXTTextProcessor(
            system="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
            "The visual content will be provided with the following format: <Image>visual content</Image>.",
            roles=("Human", "Assistant"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.SINGLE,
            sep="###",
            version="v0_mmtag",
        ),
        "llava_v1": LLaVANeXTTextProcessor(
            system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        ),
        "llava_v1_mmtag": LLaVANeXTTextProcessor(
            system="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
            "The visual content will be provided with the following format: <Image>visual content</Image>.",
            roles=("USER", "ASSISTANT"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
            version="v1_mmtag",
        ),
        "llava_llama_2": LLaVANeXTTextProcessor(
            system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
            roles=("USER", "ASSISTANT"),
            version="llama_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="<s>",
            sep2="</s>",
        ),
        # "llava_llama_3": LLaVANeXTTextProcessor(
        #     system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
        #     roles=("<|start_header_id|>user", "<|start_header_id|>assistant"),
        #     version="llama_v3",
        #     messages=[],
        #     offset=0,
        #     sep_style=SeparatorStyle.LLAMA_3,
        #     tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     tokenizer=AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"),
        #     stop_token_ids=[128009],
        # ),
        "llava_llama_2_simple": LLaVANeXTTextProcessor(
            system="Answer the questions about the visual content that the user provides.",
            roles=("USER", "ASSISTANT"),
            version="llama_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="<s>",
            sep2="</s>",
        ),
        "llava_llama_2_mmtag": LLaVANeXTTextProcessor(
            system="Answer the questions about the visual content that the user provides." "The visual content will be provided with the following format: <Image>visual content</Image>.",
            roles=("USER", "ASSISTANT"),
            version="llama_v2_mmtag",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="<s>",
            sep2="</s>",
        ),
        "llava_mistral_instruct": LLaVANeXTTextProcessor(
            system="",
            roles=("USER", "ASSISTANT"),
            version="llama_v2",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.LLAMA_2,
            sep="",
            sep2="</s>",
        ),
        "mpt": LLaVANeXTTextProcessor(
            system="""<|im_start|>system
        A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        ),
        "qwen_1_5": LLaVANeXTTextProcessor(
            system="""<|im_start|>system
        You are a helpful assistant.""",
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            version="qwen",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        ),
        "gemma_instruct": LLaVANeXTTextProcessor(
            system="", 
            roles=("<start_of_turn>user\n", "<start_of_turn>model\n"), 
            version="gemma", 
            messages=[], 
            offset=0, 
            sep_style=SeparatorStyle.GEMMA, 
            sep="<end_of_turn>\n"
        )
    }[model_name]