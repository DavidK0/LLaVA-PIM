# This script takes one argument:
#  the path to fv_albt_test_11152023
# This script will process those images and generate product descriptions for each one

import argparse
import sys
import os

import LLaVA_Product_Descriptions as LLaVA

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", default=None)
args = parser.parse_args()

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']

UPCs = UPCs[0:1]

# Load model
model_args = ["--model-path", "liuhaotian/llava-v1.5-7b", "--load-8bit"]
LLaVA.load_model(model_args)

for UPC in UPCs:
    image_paths = os.path.join(args.reference_images, UPC)
    csv_path = os.path.join(args.reference_images, UPC, f"{UPC}.csv")
    
    inference_args = ["--image-file", image_paths, "--csv_output", csv_path]
    LLaVA.inference(inference_args)