import os
import json

from utils import parse_arguments
from models import load_model
from dataset import VidHalDataset
from pipelines.inference import get_inference_pipeline

if __name__ == "__main__":
    args = parse_arguments()

    # Load model and dataset
    model, vis_processor, text_processor = load_model(args.model)
    dataset = VidHalDataset(
        args.annotations_path, args.videos_path, vis_processor, args.num_frames, load_video=(args.model != "random")
    )
    if args.options_path:
        with open(args.options_path, "r") as f:
            option_display_order = json.load(f)
    else:
        option_display_order = None

    # Load inference pipeline and run inference
    inference_pipeline = get_inference_pipeline(args.model, args.task)(
        model=model, dataset=dataset,
        num_captions=args.num_captions, 
        option_display_order=option_display_order
        # TODO: Additional arguments if any are added
    )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    inference_pipeline.run(save_path=args.save_path)
