# This is the last step in the classification pipeline.
# It takes four arguments.

import argparse
import random
import math
import json
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from utils import normalized_edit_distance, read_csv

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("reference_descriptions", type=str, action="store")
parser.add_argument("input_descriptions", type=str, action="store")
parser.add_argument("pl_candidate_csv", type=str, action="store")
#parser.add_argument("classification_output", type=str, action="store")
args = parser.parse_args()

pl_info_header, pl_info = read_csv(args.pl_candidate_csv, has_header=True)

# Get indices based on the header
upc_index = pl_info_header.index("candidate_upc")
distance_index = pl_info_header.index("candidate_distance")
label_id_index = pl_info_header.index("label_id")
code_from_user_index = pl_info_header.index("code_from_user")

label_ids = {}
for row in pl_info:
    label_id = row[pl_info_header.index("label_id")]
    if label_id not in label_ids:
        label_ids[label_id] = []
    label_ids[label_id].append(row)

# Read reference descriptions and convert to a dictionary
reference_data = [json.loads(line) for line in open(args.reference_descriptions)]
reference_descriptions = {x[0] : x[1:] for x in reference_data}

# Read input descriptions
input_descriptions = [json.loads(line) for line in open(args.input_descriptions)]

# Threshold constants
DESCRIPTION_SIMILARITY_THRESHOLD = 0.11  # Scores less than this mean the VLM descriptions are unconfident
CORE_DISTANCE_THRESHOLD = 0.14 # Distances greater than this means code from core is unconfident

def get_description_similarity(description1, description2, weights):
    """
    Returns the similarity score between two descriptions. Each description must be a list of
      strings, and the lists must be the same length.
    The matching score between two descriptions is defined as the weighted average normalized edit distance (NED)
      across all strings in the descriptions,
      where NED(str1, str2) = Levenshtein(str1, str2) / max(len(str1), len(str2))
    Capitalization is ignored.
    Returns a float.
    """
    assert(len(description1) == len(description2))
    distances = []
    # Iterate over description elements
    for i in range(len(description1)):
        # Convert to lower case and find the NED
        distance = normalized_edit_distance(description1[i].lower(), description2[i].lower())
        distances.append(distance)

    # Compute weighted average of NEDs
    normalized_weights = [weight/sum(weights) for weight in weights]
    weighted_distances = [distance*weight for distance, weight in zip(distances, normalized_weights)]
    average_distance = sum(weighted_distances) / len(weighted_distances)
    
    return average_distance

def classify(input_description, weights):
    """Classifies the given input_description. Returns two values, a UPC and the classification method).
      The classification method is either "core" or "vlm". """
    
    label_id = os.path.splitext(os.path.basename(input_description[0]))[0]
    input_description = input_description[1:]

    # Get pl candidates
    pl_candidate_upcs = [row[upc_index].rjust(12, "0") for row in label_ids[label_id]]
    pl_candidate_descriptions = [x for x in reference_descriptions if x[0] in pl_candidate_upcs]

    scores_list = {}
    # Iterate over reference images
    for candidate in label_ids[label_id]:
        candidate_upc = candidate[upc_index].rjust(12, "0")

        # Get core distance
        try:
            core_distance = float(candidate[distance_index])
        except:
            continue
        
        # Get description similarity
        # Check that we have a description for this candidate
        if candidate_upc in reference_descriptions:
            description_similarity = get_description_similarity(input_description, reference_descriptions[candidate_upc], weights)
        else:
            print(f"[WARNING] {label_id} is missing from the reference descriptions")
            description_similarity = math.inf # Block this UPC from being selected by the VLM, but not by Core
        
        scores_list[candidate_upc] = (core_distance, description_similarity)

    # If core distance is low or description similarity is high, use the UPC that core gives
    min_distance = min(scores[0] for upc, scores in scores_list.items())
    max_similarity = max(scores[1] for upc, scores in scores_list.items())
    
    if min_distance < CORE_DISTANCE_THRESHOLD or max_similarity > DESCRIPTION_SIMILARITY_THRESHOLD:
        return  min(scores_list, key=lambda x: scores_list[x][0]), "core"
    else:
        return min(scores_list, key=lambda x: scores_list[x][1]), "vlm"

def main(weights, thresholds=None):
    
    
    global DESCRIPTION_SIMILARITY_THRESHOLD, CORE_DISTANCE_THRESHOLD
    if thresholds:
        DESCRIPTION_SIMILARITY_THRESHOLD = thresholds[0]
        CORE_DISTANCE_THRESHOLD = thresholds[1]
    
    answer_key = {}
    for row in pl_info:
        answer_key[row[label_id_index]] = row[code_from_user_index].rjust(12, '0')
    
    total_correct = 0
    core_correct = [0, 0]
    vlm_correct = [0, 0]
    input_subset = input_descriptions[len(input_descriptions)//2:]
    for input_description in tqdm(input_subset, leave=False):
        predicted_UPC, classification_method = classify(input_description, weights)
        
        if label_id not in answer_key:
            raise Exception(f"{label_id} is not in the answer key")
        
        gold_UPC = answer_key[os.path.splitext(os.path.basename(input_description[0]))[0]]
        
        if classification_method == "core":
            core_correct[0] += 1
        else:
            vlm_correct[0] += 1
        
        if predicted_UPC == gold_UPC:
            total_correct += 1
            if classification_method == "core":
                core_correct[1] += 1
            else:
                vlm_correct[1] += 1
        else:
            if classification_method == "vlm":
                print(input_description, gold_UPC, predicted_UPC)
    
    total_accuracy = total_correct/len(input_subset) if len(input_subset) > 0 else 0
    print(f"Accuracy: {total_correct}/{len(input_subset)} ({total_accuracy:.1%})")
    
    core_accuracy = core_correct[1]/core_correct[0] if core_correct[0] > 0 else 0
    print(f"Core accuracy: {core_correct[1]}/{core_correct[0]} ({core_accuracy:.1%})")
    
    vlm_accuracy = vlm_correct[1]/vlm_correct[0] if vlm_correct[0] > 0 else 0
    print(f"VLM accuracy: {vlm_correct[1]}/{vlm_correct[0]} ({vlm_accuracy:.1%})")
    
    return total_correct/len(input_subset)

def optimize_weight(index, weights, step=0.03, min_val=0, max_val=2):
    best_accuracy = 0
    best_weight = weights[index]
    for w in [max_val - step * i for i in range(int((max_val - min_val) / step) + 1)]:
        weights[index] = w
        print(weights)
        
        accuracy = main(weights)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = w
    return best_weight

def grid_search(weights, step_size, start=0, stop=1):
    best_accuracy = 0
    best_thresholds = [start, start]
    num_steps = int((stop - start) / step_size) + 1  # Calculate the number of steps based on the range and step size
    accuracy_grid = np.zeros((num_steps, num_steps))

    for w1 in range(num_steps):  # Iterate from 0 to num_steps
        for w2 in range(num_steps):
            thresholds = [start + w1 * step_size, start + w2 * step_size]
            print(thresholds)
            accuracy = main(weights, thresholds)
            accuracy_grid[w1, w2] = accuracy
            if accuracy > best_accuracy:
                print("new best found")
                best_accuracy = accuracy
                best_thresholds = thresholds

    return best_thresholds, best_accuracy, accuracy_grid

def plot_accuracy_grid(accuracy_grid, step_size):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(accuracy_grid, cmap='hot', origin='lower')

    # Setting the ticks based on the step size
    num_steps = accuracy_grid.shape[0]
    tick_marks = np.arange(0, num_steps, step_size * 10)  # Adjust this for better tick mark spacing
    labels = [f"{x * step_size:.2f}" for x in tick_marks]
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.colorbar(im)
    plt.xlabel('Core Threshold')
    plt.ylabel('VLM Threshold')
    plt.title('Grid Search Accuracy Heatmap')
    plt.savefig("plot.jpg", bbox_inches='tight')
    
if __name__ == "__main__":
    weights = [1 for x in range(len(input_descriptions[0]) - 1)]
    #weights = [1.44, 0.0, 0.03, 0.72, 0.63, 0.09, 0.0, 0.0, 0.45, 0.39, 0.51, 0.06]
    
    accuracy = main(weights)
    
    # Use this to classify many times to find the ideal thresholds
    #step_size = .01
    #best_thresholds, best_accuracy, accuracy_grid = grid_search(weights, step_size=step_size,start=.05, stop = .2)
    #print(f"Best thresholds: {best_thresholds}, Best accuracy: {best_accuracy}")
    #plot_accuracy_grid(accuracy_grid, step_size)
    
    # Use this to classify many times to find the ideal weights
    #for i in range(len(weights)):
    #    weights[i] = optimize_weight(i, weights)
    #print(f"Best weights: {weights}")
    #accuracy = main(weights)