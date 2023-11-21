# This is the last step in the classification pipeline

# Ignores case

import argparse
import sys
import os

from utils import levenshtein_distance, read_csv

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("reference_descriptions", type=str, action="store")
parser.add_argument("input_descriptions", type=str, action="store")
parser.add_argument("classification_output", type=str, action="store")
parser.add_argument("--answer-key", type=str, action="store")
args = parser.parse_args()

# Read reference descriptions
_, reference_descriptions = read_csv(args.reference_descriptions, has_header=True)
header, input_descriptions = read_csv(args.input_descriptions, has_header=True)

label_classifications = {}
for input_description in tqdm(input_descriptions, position=2):
    total_distances = []
    for reference_description in reference_descriptions:
        distances = []
        for i in range(1, len(header)):
            
            # Convert to lower case
            answer = input_description[i].lower()
            reference_answer = reference_description[i].lower()
            
            distance = levenshtein_distance(answer, reference_answer)
            #print(answer, reference_answer, distance)
            distances.append(distance)
        total_distance = sum(distances)
        total_distances.append(total_distance)
        #print(reference_description[0], distances, total_distance)
    #print(total_distances)
    sorted_indices = sorted(range(len(total_distances)), key=lambda x: total_distances[x])
    #print(sorted_indices)
    label_id = input_description[0][:-4]
    label_classifications[label_id] = reference_descriptions[sorted_indices[0]][0][1:] # Cut off the leading '0'
    #print(label_id, )

if not args.answer_key:
    sys.exit()

header, answer_data = read_csv(args.answer_key, has_header=True)
answer_key = {}
for row in answer_data:
    answer_key[row[2]] = row[5]

total_correct = 0
for label_id, predicted_UPC in label_classifications.items():
    if label_id not in answer_key:
        raise Exception(f"{label_id} is not in the answer key")
    if predicted_UPC == answer_key[label_id]:
        total_correct += 1

accuracy = total_correct/len(label_classifications)
print(f"Total correct: {total_correct}/{len(label_classifications)} ({accuracy:.2%})")