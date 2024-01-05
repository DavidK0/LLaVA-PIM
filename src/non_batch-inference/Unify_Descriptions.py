# This script is used to combine the non_batch-inference results into a single .jsonl file.
# It takes two arguments:
#   batched_results: this is the output from Process_Images.py
#   output: this is the output .jsonl holding the aggregatd product descriptions

import argparse
import json
import time
import csv
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import find_min_average_distance_word, read_csv

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("product_descriptions", type=str, action="store", help="The path to fv_ablt_test")
parser.add_argument("jsonl_output", type=str, action="store", help="The name of the output to save to. Defaults to run-name", default=None)
args = parser.parse_args()

# Read the .csv files
UPCs = {}
for file in os.listdir(args.product_descriptions):
    if not ".csv" in file:
        continue
    header, data = read_csv(os.path.join(args.product_descriptions, file), has_header=True)
    UPC = file.split(".")[0]
    UPCs[UPC] = [row[1:] for row in data]

if all([UPCs[UPC] == None for UPC in UPCs]):
    raise Exception(f"No run with the name {args.run_name} found in {args.reference_images}")

# De-collate the decsriptions
UPCs_unified_decscriptions = {}
start_time = time.time()
for UPC in tqdm(UPCs):
    product_descriptions = UPCs[UPC]
    
    if not product_descriptions:
        continue
    
    UPCs_unified_decscriptions[UPC] = [UPC]
    
    for i in range(len(product_descriptions[0])):
        # Skip the second to last item:
        if i == 9:
            continue
        
        # Limit the length of each string
        strings = [x[i].lower()[:100] for x in product_descriptions]
        
        selected_word = find_min_average_distance_word(strings)
        
        UPCs_unified_decscriptions[UPC].append(selected_word)
    
    # Save the results
    with open(args.jsonl_output, "a") as file:
            file.write(json.dumps(UPCs_unified_decscriptions[UPC]) + "\n")

runtime = time.time() - start_time
print(f"Runtime: {runtime:.2f}")



header = ["UPC"] + header[1:]