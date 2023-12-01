# This script takes two arguments, the model path and the image(s) path.
# It optionall takes an output path

import argparse
import sys
import os
import LLaVA_Product_Descriptions as LLaVA

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, action="store", default=None, required=True)
parser.add_argument("--images-path", type=str, action="store", default=None, required=True)
parser.add_argument("--csv-output", type=str, action="store", default=None, required=True)
args = parser.parse_args()

image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}  # Add more if needed
folders_with_images = set()

for root, dirs, files in os.walk(args.images_path):
    # Skip hidden directories
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            folders_with_images.add(root)
            break  # Once an image is found, move to the next directory

# Load model
model_args = ["--model-path", args.model_path, "--load-8bit"]
LLaVA.load_model(model_args)

# Run inference
imageQAs = []
for folder in folders_with_images:
    inference_args = ["--image-file", folder]
    imageQAs.extend(LLaVA.inference(inference_args))

LLaVA.save_results_to_csv(args.csv_output,imageQAs)