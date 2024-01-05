# This script is used to combine the batch-inference results into a single .jsonl file.
# It takes two arguments:
#   batched_results: this is the output from Make_Questions_jsonl_long.py
#   output: this is the output .jsonl holding the aggregatd product descriptions

import argparse
import json
import time
import csv
import sys
import os

from utils import find_min_average_distance_word
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("batched_results", action="store", type=str, help="This is the output from Make_Questions_jsonl_long.py")
parser.add_argument("output", action="store", type=str, help="This is the output .jsonl holding the aggregatd product descriptions")
args = parser.parse_args()

start_time = time.time()

# Read batched resutls
answer_files = sorted([os.path.join(args.batched_results,file) for file in os.listdir(args.batched_results) if ".jsonl" in file and "answer" in file])
question_files = sorted([os.path.join(args.batched_results,file) for file in os.listdir(args.batched_results) if ".jsonl" in file and "question" in file])

UPCs = {}

# Read questions file (for the image names)
data = [json.loads(line) for line in open(question_files[0])]
image_names = [line["image"] for line in data]

for answer_file in tqdm(answer_files):
    answers = [json.loads(line) for line in open(answer_file)]
    
    # Read answers files
    current_answers = []
    current_UPC = image_names[0].split("_")[0]
    for image_name, answer in zip(image_names, answers):

        new_UPC = image_name.split("_")[0]

        # Check to add a new UPC
        if new_UPC not in UPCs:
            UPCs[new_UPC] = []

        if new_UPC != current_UPC:
            # Find the best word
            selected_word = find_min_average_distance_word(current_answers)
            
            #print(current_answers, selected_word)
            UPCs[current_UPC].append(selected_word)


            # Reset the current information
            current_UPC = new_UPC
            current_answers = []
        #else:
        # Convert to lower case, restrict the length, and append to the list
        current_answers.append(answer["text"].lower()[:100])

# Save the results
with open(args.output, "w") as file:
    for UPC, data in UPCs.items():
        file.write(json.dumps([UPC] + data) + "\n")
        
end_time = time.time()
print(f"Runtime: {end_time - start_time}")