# This is the last step in the classification pipeline.
# It takes four arguments.

import argparse
import sys
import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from utils import normalized_edit_distance, read_csv

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("reference_descriptions", type=str, action="store")
parser.add_argument("input_descriptions", type=str, action="store")
parser.add_argument("classification_output", type=str, action="store")
parser.add_argument("--answer-key", type=str, action="store")
args = parser.parse_args()

pl_info_header, pl_info = read_csv(args.answer_key, has_header=True)

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
_, reference_descriptions = read_csv(args.reference_descriptions, has_header=True)
reference_descriptions = {x[0] : x[1:] for x in reference_descriptions}

# Read input descriptions
_, input_descriptions = read_csv(args.input_descriptions, has_header=True)

# Threshold constants
DESCRIPTION_SIMILARITY_THRESHOLD = .27 # Scores less than this mean the VLM descriptions are unconfident
CORE_DISTANCE_THRESHOLD = .18 # Distances greater than this means code from core is unconfident

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
    """Classifies the given input_description"""
    
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
            description_similarity = math.inf # Block this UPC from being selected by the VLM, but not by Core
        
        scores_list[candidate_upc] = (core_distance, description_similarity)

    # If core distance is low or description similarity is high, use the UPC that core gives
    min_distance = min(scores[0] for upc, scores in scores_list.items())
    max_similarity = min(scores[0] for upc, scores in scores_list.items())
    
    if min_distance < CORE_DISTANCE_THRESHOLD or max_similarity > DESCRIPTION_SIMILARITY_THRESHOLD:
        return  min(scores_list, key=lambda x: scores_list[x][0])
    else:
        return min(scores_list, key=lambda x: scores_list[x][1])

def main(w=None):
    weights = [0.3, 0.4, 0.1, 0.0, 0.8, 1.0, 1.4, 0.1, 0.9, 1.2, 0.8, 2.0]
    
    global DESCRIPTION_SIMILARITY_THRESHOLD, CORE_DISTANCE_THRESHOLD
    if w:
        DESCRIPTION_SIMILARITY_THRESHOLD = w[0]
        CORE_DISTANCE_THRESHOLD = w[1]
    
    answer_key = {}
    for row in pl_info:
        answer_key[row[label_id_index]] = row[code_from_user_index].rjust(12, '0')
    
    total_correct = 0
    for input_description in tqdm(input_descriptions, leave=False):
        predicted_UPC = classify(input_description, weights)
        
        if label_id not in answer_key:
            raise Exception(f"{label_id} is not in the answer key")
        
        gold_answer = answer_key[os.path.splitext(os.path.basename(input_description[0]))[0]]
        #if gold_answer not in accuracy_per_UPC:
        #    accuracy_per_UPC[gold_answer] = [0,0, []]
        #accuracy_per_UPC[gold_answer][1] += 1
        
        if predicted_UPC == gold_answer:
            total_correct += 1
    
    accuracy = total_correct/len(input_descriptions)
    print(f"Accuracy: {accuracy:.1%}")
    return accuracy

def optimize_weight(index, weights, step=0.1, min_val=0, max_val=1):
    best_accuracy = 0
    best_weight = weights[index]
    for w in [min_val + step * i for i in range(int((max_val - min_val) / step) + 1)]:
        weights[index] = w
        print(weights)
        
        accuracy = main(weights)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = w
    return best_weight

def grid_search(step_size, start=0, stop=1):
    best_accuracy = 0
    best_weights = [start, start]
    num_steps = int((stop - start) / step_size) + 1  # Calculate the number of steps based on the range and step size
    accuracy_grid = np.zeros((num_steps, num_steps))

    for w1 in range(num_steps):  # Iterate from 0 to num_steps
        for w2 in range(num_steps):
            weights = [start + w1 * step_size, start + w2 * step_size]
            print(weights)
            accuracy = main(weights)
            accuracy_grid[w1, w2] = accuracy
            if accuracy > best_accuracy:
                print("new best found")
                best_accuracy = accuracy
                best_weights = weights

    return best_weights, best_accuracy, accuracy_grid

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
    #weights = [1 for x in range(len(header) - 1)]
    #weights = [0.0, 0.2, 0.0, 0.6, 0.4, 0.6, 0.4] # Vote 7q
    #weights = [0.1, 0.6, 0.0, 1.3, 0.5, 0.7, 0.3] # Levenstein 7q
    #weights = [0.8, 0.2, 0.6, 0.6, 1.8, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0] # Vote 12q
    #weights = [0.0, 0.4, 0.7, 1.1, 1.0, 1.0, 0.0, 1.0, 1.0, 0.4, 0.7, 0.5] # Levenstein 12q
    #weights = [0.3, 0.4, 0.1, 0.0, 0.8, 1.0, 1.4, 0.1, 0.9, 1.2, 0.8, 2.0] # Levenstein 12q candidates
    
    #weights = [1,1]
    accuracy = main()
    
    #step_size = .03
    #best_weights, best_accuracy, accuracy_grid = grid_search(step_size=step_size,start=.5/4., stop = 1./3.)
    #print(f"Best weights: {best_weights}, Best accuracy: {best_accuracy}")
    #plot_accuracy_grid(accuracy_grid, step_size)
    
    # Use this to classify many times to find the ideal weights
    #for i in range(len(weights)):
    #    weights[i] = optimize_weight(i, weights)
