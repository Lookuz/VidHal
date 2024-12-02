import os
import json

from utils import parse_arguments
from models import load_model
from dataset import VidHalDataset
from pipelines.inference import get_inference_pipeline

if __name__ == "__main__":
    args = parse_arguments()

    # Load model and dataset
    model, vis_processor, text_processor = load_model(
        args.model,
        model_path=args.model_path, config_path=args.config_path,
        num_frames=args.num_frames, load_4bit=args.load_4bit, load_8=args.load_8bit,
        # LLaVa-NeXT-Video override parameters
        mm_spatial_pool_mode=args.mm_spatial_pool_mode, 
        mm_newline_position=args.mm_newline_position,
        mm_pooling_position=args.mm_pooling_position,
    )
    dataset = VidHalDataset(
        args.annotations_path, args.videos_path, vis_processor, args.num_frames, load_video=(args.model != "random")
    )
    if args.options_path:
        with open(args.options_path, "r") as f:
            option_display_order = json.load(f)
    else:
        option_display_order = None

    api_key = args.api_key
    if api_key is not None and os.path.isfile(api_key):
        with open(api_key, "r") as f:
            api_key = f.readlines()[0].strip()
    # Load inference pipeline and run inference
    inference_pipeline = get_inference_pipeline(args.model, args.task)(
        model=model, dataset=dataset,
        vis_processor=vis_processor, text_processor=text_processor,
        model_path=args.model_path,
        num_captions=args.num_captions, 
        option_display_order=option_display_order,
        api_key=api_key
        # TODO: Additional arguments if any are added
    )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    inference_pipeline.run(save_path=args.save_path)
