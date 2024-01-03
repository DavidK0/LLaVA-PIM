# This script is for taking the output of of LLaVA batch inference (answers.jsonl) and converting
#   it to the .csv format I had been using for non-batched inference.


import argparse
import json
import sys
import csv
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("answers_dir", type=str, action="store")
parser.add_argument("output_dir", type=str, action="store")
args = parser.parse_args()

answer_files = [os.path.join(args.answers_dir, file) for file in os.listdir(args.answers_dir) if "answer" in file]
question_files = [os.path.join(args.answers_dir, file) for file in os.listdir(args.answers_dir) if "question" in file]

# Maps from question index to product description
question_index_to_description = {}
# Maps from image name to question index
image_name_to_question_index = {}
# Maps from UPC to image name
upc_to_image_name = {}

# Setup up the maps
for question_file in tqdm(question_files[:1]):
    with open(question_file) as file:
        for line in file:
            data = json.loads(line)
            question_index_to_description[data["question_id"]] = []
            image_name_to_question_index[data["image"]] = data["question_id"]
            
            UPC = data["image"].split("_")[0]
            if UPC not in upc_to_image_name:
                upc_to_image_name[UPC] = []
            upc_to_image_name[UPC].append(data["image"])
            break

print(f"Number of images:  {len(question_index_to_description)}")

# Read the product descriptions from the answers.jsonl
for answer_file in tqdm(sorted(answer_files)):
    with open(answer_file) as file:
        for line in file:
            data = json.loads(line)
            question_index_to_description[data["question_id"]].append(data["text"])
            break

# Make the output directory
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Save the descriptions to .csv's
for UPC, image_name_list in tqdm(upc_to_image_name.items()):
    csv_path = os.path.join(args.output_dir, f"{UPC}.csv")
    with open(csv_path, "w") as file:
        writer = csv.writer(file)
        #print(len(answer_files))
        header = ["image_name"] + [str(x + 1) for x in range(len(answer_files))]
        writer.writerow(header)
        for image_name in image_name_list:
            writer.writerow([image_name] + question_index_to_description[image_name_to_question_index[image_name]])
            
