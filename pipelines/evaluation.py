import string
import random
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from ..dataset import VidHalDataset

class EvaluationPipeline:
    def __init__(
        self, 
        predictions : dict, # dict of video_id -> predicted responses
        dataset : VidHalDataset, 
        option_display_order : dict = None, # Optional argument specifying the pre-defined randomization seed for input caption display order
        *args, **kwargs
    ):
        self.predictions = predictions
        self.dataset = dataset
        self.option_display_order = option_display_order if option_display_order is not None else self.generate_display_order()

    def generate_display_order(self):
        option_display_order = {}
        for i in tqdm(range(len(self.dataset))):
            example = self.dataset[i]
            video_id, captions = example["video_id"], example["captions"]
            caption_keys = list(captions.keys())
            random.shuffle(caption_keys)

            # Assign permuted options (A, B, C) to actual order (1, 2, 3)
            option_prefix = list(string.ascii_uppercase)
            option_to_rank = OrderedDict(
                {option_prefix[i]: option for i, option in enumerate(caption_keys)}
            )

            option_display_order[video_id] = option_to_rank

        return option_to_rank

    def evaluate(self):
        raise NotImplementedError
    
class VidHalMCQAEvaluationPipeline(EvaluationPipeline):
    def __init__(self, predictions, dataset, option_display_order = None, *args, **kwargs):
        super().__init__(predictions, dataset, option_display_order, *args, **kwargs)

    def evaluate(self):
        accuracy, total = {"overall" : 0}, {"overall" : 0}
        for i in tqdm(range(len(self.dataset))):
            example = self.dataset[i]
            video_id, captions, aspect = example["video_id"], example["captions"], example["aspect"]
            if aspect not in accuracy: 
                accuracy[aspect] = 0
            if aspect not in total:
                total[aspect] = 0
            
            option_to_rank = self.option_display_order[video_id]
            answer = {v : k for k, v in option_to_rank.items()}["1"]
            prediction, answer_phrase = self.predictions[video_id], captions["1"]
            is_correct = (
                int(prediction == answer) or (answer_phrase.lower().strip(".")) in prediction # Account for situation where VLLM response is caption instead of option
            )

            for key in [aspect, "overall"]:
                accuracy[key] += is_correct
                total[key] += 1

        for aspect in accuracy:
            accuracy[aspect] = accuracy[aspect] / total[aspect]

        return accuracy

class VidHalCaptionOrderingEvaluationPipeline(EvaluationPipeline):
    def __init__(
        self, predictions, dataset, option_display_order = None, 
        num_captions : int = 3,
        i_normalize : bool = True, # NDCG normalization factors
        r_normalize : bool = True,
        *args, **kwargs):
        super().__init__(predictions, dataset, option_display_order, *args, **kwargs)

        self.num_captions = num_captions
        self.i_normalize_value = self.compute_dcg(list(range(1, self.num_captions + 1))) if i_normalize else None
        self.r_normalize_value = self.compute_dcg(reversed(list(range(1, self.num_captions + 1)))) if r_normalize else None

    def compute_dcg(self, order):
        """
        Takes in a sequence of numbers representing the hallucination extent
        """
        relevance_scores = [self.num_captions + 1 - int(self.option_display_order[i]) for i in order]
        return sum([score / np.log2(i + 2) for i, score in enumerate(relevance_scores)])        
    
    def compute_ndcg(self, order, option_to_rank):
        """
        Takes in a sequence of options representing the captions
        """
        # NOTE: Ignore partial ordering for now. To revisit later.
        if len(order) < self.num_captions: return 0.
        order = [option_to_rank[x] for x in order] 

        ndcg = self.compute_dcg(order)
        if self.r_normalize_value is not None:
            ndcg -= self.r_normalize_value
        if self.i_normalize_value is not None:
            ndcg = ndcg / (self.i_normalize_value - self.r_normalize_value) if self.i_normalize_value is not None else ndcg / self.i_normalize_value

        return ndcg
    
    def evaluate(self):
        ndcg, total, order_prediction_frequency = {"overall" : 0}, {"overall" : 0}, {}
        for i in tqdm(range(len(self.dataset))):
            example = self.dataset[i]
            video_id, aspect = example["video_id"], example["aspect"]
            if aspect not in ndcg: 
                ndcg[aspect] = 0
            if aspect not in total:
                total[aspect] = 0

            option_to_rank = self.option_display_order[video_id]
            # Predictions expected to be either in comma separated string form (e.g 'A, B, C') or list form (e.g. ['A', 'B', 'C'])
            prediction = self.predictions(video_id)
            if not isinstance(prediction, list):
                prediction_key = prediction
                prediction = [x.strip() for x in prediction.split(",")]
            else:
                prediction_key = ", ".join(prediction)
            
            if prediction_key in order_prediction_frequency:
                order_prediction_frequency[prediction_key] += 1
            else:
                order_prediction_frequency[prediction_key] = 1

            ndcg_ = self.compute_ndcg(prediction, option_to_rank)
            for key in [aspect, "overall"]:
                ndcg[key] += ndcg_
                total[key] += 1

            for aspect in ndcg:
                ndcg[aspect] = ndcg[aspect] / total[aspect]

        return {"ndcg" : ndcg, "frequency" : order_prediction_frequency}
