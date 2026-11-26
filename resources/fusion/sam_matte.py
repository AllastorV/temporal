"""Generate a point-guided subject matte for Temporal from a local SAM 2.1 checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


def segment(image, prompts, model, processor, device):
    matte = np.zeros((image.height, image.width), dtype=np.float32)
    for prompt in prompts:
        left, top, right, bottom = prompt.get("crop", (0, 0, image.width, image.height))
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("Crop is outside the image.")
        points = prompt["points"]
        if not any(label == 1 for _, _, label in points) or any(
            not (left <= x < right and top <= y < bottom and label in (0, 1))
            for x, y, label in points
        ):
            raise ValueError("Each prompt needs an in-bounds foreground point and labels of 0 or 1.")
        inputs = processor(
            images=image.crop((left, top, right, bottom)),
            input_points=[[[[x - left, y - top] for x, y, _ in points]]],
            input_labels=[[[label for _, _, label in points]]],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            result = model(**inputs, multimask_output=False)
            logits = processor.post_process_masks(
                result.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=False
            )[0][0, 0]
        # Keep a soft coverage transition around the model's zero-logit contour.
        coverage = torch.sigmoid(2 * logits).numpy()
        coverage = np.clip((coverage - 0.05) / 0.9, 0.0, 1.0)
        region = matte[top:bottom, left:right]
        np.maximum(region, coverage, out=region)
    return matte


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("prompts", type=Path, help="JSON list of point groups with optional crop bounds")
    parser.add_argument("output", type=Path, help="Grayscale PNG. White protects the subject.")
    parser.add_argument("--model", type=Path, required=True, help="Local SAM 2.1 model directory")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not prompts:
        parser.error("At least one point group is required.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Sam2Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam2Model.from_pretrained(args.model, local_files_only=True).to(device).eval()
    with Image.open(args.source) as source:
        matte = segment(source.convert("RGB"), prompts, model, processor, device)
    Image.fromarray(np.rint(matte * 255).astype(np.uint8)).save(args.output)


if __name__ == "__main__":
    main()
