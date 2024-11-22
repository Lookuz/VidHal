import os
import json

from utils import parse_arguments
from dataset import VidHalDataset
from pipelines.evaluation import VidHalMCQAEvaluationPipeline, VidHalCaptionOrderingEvaluationPipeline

if __name__ == "__main__":
    args = parse_arguments()

    # Load dataset
    dataset = VidHalDataset(
        args.annotations_path, args.videos_path, vis_processor=None, num_frames=args.num_frames, load_video=(args.model != "random")
    )
    if args.options_path:
        with open(args.options_path, "r") as f:
            option_display_order = json.load(f)
    else:
        option_display_order = None

    # Load predictions
    assert args.predictions_path is not None, "Path to generated responses must be provided when running evaluation!"
    with open(args.predictions_path, "r") as f:
        predictions = json.load(f)

    # Load evaluation pipeline and run evaluation
    evaluation_pipeline = {
        "mcqa" : VidHalMCQAEvaluationPipeline, 
        "naive_ordering" : VidHalCaptionOrderingEvaluationPipeline,
        "relative_ordering" : VidHalCaptionOrderingEvaluationPipeline
    }[args.task](
        predictions, dataset,
        option_display_order=option_display_order,
        num_captions=args.num_captions
    )
    results = evaluation_pipeline.evaluate()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(results, f, indent=4)
