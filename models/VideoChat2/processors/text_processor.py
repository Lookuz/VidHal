import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from enum import auto, Enum
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Chat logic from VideoChat2
"""
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


def get_prompt(conv, answer_prompt=None):
    ret = conv.system + conv.sep
    for i, (role, message) in enumerate(conv.messages):
        if message:
            ret += role + ": " + message
            if i < len(conv.messages) - 1 and answer_prompt is not None:
                ret += conv.sep
        else:
            ret += role + ":"
    return ret


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

class VideoChat2ChatProcessor:
    def __init__(self, model="llama", device=device):
        self.device = device
        stop_words_ids = {
            "llama" : [torch.tensor([835]).to(self.device), torch.tensor([2277, 29937]).to(self.device)],
            "mistral" : [torch.tensor([2]).to("cuda:0"), torch.tensor([29871, 2]).to("cuda:0")],
            "phi" : [torch.tensor([32000]).to("cuda:0"), torch.tensor([32007]).to("cuda:0")]
        }[model]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text + '\n'])
        return conv

    def answer(
        self, model, video_embs, conv=None, 
        max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9, do_sample=False, num_return_sequences=1,
        repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None):

        conv.messages.append([conv.roles[1], answer_prompt])
        embs = self.get_context_emb(
            model=model, 
            conv=conv, 
            video=video_embs, 
            answer_prompt=answer_prompt
        )

        outputs = model.language_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_return_sequences
        )
        output_ids = outputs[0]
        if output_ids[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_ids = output_ids[1:]
        if output_ids[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_ids = output_ids[1:]
        if output_ids.shape[0] <= 1:
            output_text = model.lm_tokenizer.decode(output_ids, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            conv.messages[-1][1] = output_text
        # Multiple responses
        else:
            output_text = model.lm_tokenizer.batch_decode(output_ids, add_special_tokens=False)
            output_text = [x.split("###")[0].split('Assistant:')[-1].strip() for x in output_text]
            conv.messages[-1][1] = "\n".join(output_text)

        return output_text, output_ids.cpu().numpy(), conv    

    def get_context_emb(self, model, conv, video, answer_prompt=None):
        prompt = get_prompt(conv, answer_prompt=answer_prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(video) + 1, "Unmatched numbers of visual placeholders and videos."
        with torch.no_grad():
            seg_tokens = [
                model.lm_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [model.language_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], video) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
