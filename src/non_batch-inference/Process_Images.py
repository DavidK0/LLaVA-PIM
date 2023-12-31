# This script is the main script for processing images into product descriptions.
# It takes two arguments:
#   reference_images: this is a folder a folder of images of products
#   output_dir: this folder will be created and the resulting product descriptions will be placed there

import argparse
import time
import sys
import os

from tqdm import tqdm

import LLaVA_Product_Descriptions_long as LLaVA
# Use this if you don't want the model to see answers to previous questions
# import LLaVA_Product_Descriptions_short as LLaVA

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", default=None, help="This is a folder a folder of images of products")
parser.add_argument("output_dir", type=str, action="store", default=None, help="This folder will be created and the resulting product descriptions will be placed there")
parser.add_argument("--model-path", type=str, action="store", default="liuhaotian/llava-v1.5-13b")
parser.add_argument("--upc-whitelist", type=str, action="store", required=False, default=None, help="An optional textfile of allowed UPCs. Useful when splitting a large task across multiple machines")
args = parser.parse_args()

start_time = time.time()

# Check if output exists
if os.path.exists(args.output_dir):
    print(f"{args.output_dir} already exits. Existing product descriptions will be skipped and not overridden")
else:
    print(f"Creating {args.output_dir}")
    os.mkdir(args.output_dir)

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']

# Read whitelist
upc_whitelist = []
if args.upc_whitelist:
    with open(args.upc_whitelist) as file:
        upc_whitelist = [line.strip() for line in file]
    
    # Skip UPCs that are not in the whitelist
    UPCs = [UPC for UPC in UPCs if UPC in upc_whitelist]

# Skip UPCs for which a product description .csv already exists
UPCs = [UPC for UPC in UPCs if not os.path.isfile(os.path.join(args.output_dir, f"{UPC}.csv"))]

# Load model
model_args = ["--model-path", args.model_path, "--load-8bit"]
inference_parts = LLaVA.load_model(model_args)

# Run inference on each UPC
for UPC_index, UPC in enumerate(tqdm(UPCs, position=2)):
    progress = (UPC_index + 1) / len(UPCs)
    
    image_paths = os.path.join(args.reference_images, UPC)
    csv_path = os.path.join(args.output_dir, f"{UPC}.csv")

    inference_args = ["--image-file", image_paths, "--csv_output", csv_path]
    LLaVA.inference(inference_args, *inference_parts)

end_time = time.time()

print(f"Runtime: {end_time - start_time} secs")