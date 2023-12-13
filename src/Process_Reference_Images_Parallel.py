import argparse
import sys
import os
import multiprocessing
import time
from functools import partial

from tqdm import tqdm

import LLaVA_Product_Descriptions as LLaVA

# Global variable for storing the model in each process
model = None
inference_parts = None

def init_process(model_path):
    global model, inference_parts
    
    model_args = ["--model-path", model_path, "--load-4bit"]
    inference_parts = LLaVA.load_model(model_args)

def process_UPC(UPC, reference_images, output_dir):
    global model, inference_parts

    image_paths = os.path.join(reference_images, UPC)
    csv_path = os.path.join(output_dir, f"{UPC}.csv")
    
    if os.path.isfile(csv_path):
        return

    inference_args = ["--image-file", image_paths, "--csv_output", csv_path]
    LLaVA.inference(inference_args, *inference_parts)  # Assuming the model is needed here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_images", type=str, action="store", default=None)
    parser.add_argument("--output-dir", type=str, action="store", default=None, required=True)
    parser.add_argument("--model-path", type=str, action="store", default="liuhaotian/llava-v1.5-13b")
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

    # Set up multiprocessing pool
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=2, initializer=init_process, initargs=(args.model_path,))

    # Run inference on each UPC in parallel
    process_UPC_with_args = partial(process_UPC, reference_images=args.reference_images, output_dir=args.output_dir)
    for _ in tqdm(pool.imap_unordered(process_UPC_with_args, UPCs), total=len(UPCs), position=2):
        pass

    pool.close()
    pool.join()
    
    end_time = time.time()
    print(f"Runtime: {end_time - start_time} secs")