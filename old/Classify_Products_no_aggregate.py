import argparse
import csv
import os
import sys
from utils import most_common_string, find_min_average_distance_word, read_csv, save_csv, levenshtein_distance
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", help="The path to fv_ablt_test")
parser.add_argument("run_name", type=str, action="store", help="The name of the run from which the descriptions came")


parser.add_argument("input_descriptions", type=str, action="store")
#parser.add_argument("classification_output", type=str, action="store")
parser.add_argument("--answer-key", type=str, action="store")

args = parser.parse_args()

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']
UPCs = dict.fromkeys(UPCs, None)

header, input_descriptions = read_csv(args.input_descriptions, has_header=True)

# Read the .csv files
for UPC in UPCs:
    csv_path = os.path.join(args.reference_images, UPC, f"{args.run_name}_{UPC}.csv")
    if os.path.isfile(csv_path):
        header, data = read_csv(csv_path, has_header=True)
        UPCs[UPC] = [row for row in data]

if all([UPCs[UPC] == None for UPC in UPCs]):
    raise Exception(f"No run with the name {args.run_name} found in {args.reference_images}")

def classify(weights):
    global header
    label_classifications = {}
    # Iterate over input images
    for input_description in tqdm(input_descriptions, position=2):
        label_id = input_description[0][:-4]
        average_distances = []
        # Iterate over reference images
        for UPC in UPCs:
            total_distances = []
            for reference_description in UPCs[UPC]:
                distances = []
                #print(len(header),header)
                for i in range(1, len(header)):

                    # Convert to lower case
                    answer = input_description[i].lower()
                    #print(i, reference_description,input_description)
                    reference_answer = reference_description[i].lower()

                    # Find the distance for this answer
                    distance = levenshtein_distance(answer, reference_answer)
                    distances.append(distance)

                # Compute weighted sum of distances
                total_distance = sum([distance*weight for distance, weight in zip (distances, weights)])
                total_distances.append(total_distance)
            average_distances.append(sum(total_distances)/len(total_distances))

        # Find the closes matching description
        sorted_indices = sorted(range(len(average_distances)), key=lambda x: average_distances[x])
        #print( UPCs[UPC], len( UPCs[UPC]), sorted_indices[0])
        #print(list(UPCs.keys())[sorted_indices[0]])
        label_classifications[label_id] = list(UPCs.keys())[sorted_indices[0]]

    if not args.answer_key:
        sys.exit()

    _, answer_data = read_csv(args.answer_key, has_header=True)
    answer_key = {}
    for row in answer_data:
        answer_key[row[2]] = row[5].rjust(12, '0')

    total_correct = 0
    accuracy_per_UPC = {}
    for label_id, predicted_UPC in label_classifications.items():
        if label_id not in answer_key:
            raise Exception(f"{label_id} is not in the answer key")
        
        gold_answer = answer_key[label_id]
        if gold_answer not in accuracy_per_UPC:
            accuracy_per_UPC[gold_answer] = [0,0, []]
        accuracy_per_UPC[gold_answer][1] += 1
        
        if predicted_UPC == gold_answer:
            total_correct += 1
            accuracy_per_UPC[gold_answer][0] += 1
        accuracy_per_UPC[gold_answer][2] += [(label_id, predicted_UPC, predicted_UPC == gold_answer)]

    accuracy = total_correct/len(label_classifications)
    print(f"Total correct: {total_correct}/{len(label_classifications)} ({accuracy:.2%})")
    
    #for UPC in accuracy_per_UPC:
    #    num = accuracy_per_UPC[UPC][0]
    #    den = accuracy_per_UPC[UPC][1]
    #    print(f"UPC {UPC}: {num}/{den} ({(num/den):.2%})")
    #    for x in accuracy_per_UPC[UPC][2]:
    #        print(x)
    
    return accuracy

def optimize_weight(index, weights, step=0.1, min_val=0, max_val=2):
    best_accuracy = 0
    best_weight = weights[index]
    for w in [min_val + step * i for i in range(int((max_val - min_val) / step) + 1)]:
        weights[index] = w
        print(weights)
        accuracy = classify(weights)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = w
    return best_weight

if __name__ == "__main__":
    weights = [1 for x in range(len(header) - 1)]
    weights = [0.0, 0.4, 0.0, 0.7, 0.5, 0.7, 0.4] # 7q
    
    #classify(weights)
    #sys.exit()
    
    for i in range(len(weights)):
        weights[i] = optimize_weight(i, weights, step=.4)
        print(f"Optimal weight for element {i}: {weights[i]}")
    accuracy = classify(weights)
    print(f"Final accuracy with optimized weights: {accuracy}")