# This script takes three arguments.
# This script will process those images and generate product descriptions for each one.
# The product discriptions will be placed in .csv files in fv_albt_test_11152023

import argparse
import sys
import os

import LLaVA_Product_Descriptions as LLaVA

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", default=None)
parser.add_argument("--output-dir", type=str, action="store", default=None, required=True)
parser.add_argument("--model-path", type=str, action="store", default="liuhaotian/llava-v1.5-13b")
args = parser.parse_args()

# Check if output exists
if os.path.exists(args.output_dir):
    raise Exception(f"Folder '{args.output_dir}' already exists.")
else:
    print(f"Creating {args.output_dir}")
    os.mkdir(args.output_dir)

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']

# Load model
model_args = ["--model-path", args.model_path, "--load-8bit"]
LLaVA.load_model(model_args)

# Run inference on each UPC
for UPC_index, UPC in enumerate(UPCs):
    progress = (UPC_index + 1) / len(UPCs)
    print(f"Processing Reference Images: {progress:.0%}")
    
    image_paths = os.path.join(args.reference_images, UPC)
    csv_path = os.path.join(args.output_dir, f"{UPC}.csv")
    
    inference_args = ["--image-file", image_paths, "--csv_output", csv_path]
    LLaVA.inference(inference_args)