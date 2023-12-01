# This script takes one argument:
#  the path to fv_albt_test_11152023
# This script unifies the product descriptions contained in that folders

import argparse
import csv
import os
from utils import most_common_string, find_min_average_distance_word, read_csv, save_csv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", help="The path to fv_ablt_test")
parser.add_argument("--run-name", type=str, action="store", help="The name of the run from which the descriptions came", required=True)
parser.add_argument("--csv-output", type=str, action="store", help="The name of the output to save to. Defaults to run-name", default=None)
args = parser.parse_args()

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']
UPCs = dict.fromkeys(UPCs, None)

# Read the .csv files
for UPC in UPCs:
    csv_path = os.path.join(args.reference_images, UPC, f"{args.run_name}_{UPC}.csv")
    if os.path.isfile(csv_path):
        header, data = read_csv(csv_path, has_header=True)
        UPCs[UPC] = [row[1:] for row in data]

if all([UPCs[UPC] == None for UPC in UPCs]):
    raise Exception(f"No run with the name {args.run_name} found in {args.reference_images}")

# De-collate the decsriptions
UPCs_unified_decscriptions = {}
for UPC in tqdm(UPCs, position=0):
    product_descriptions = UPCs[UPC]
    if not product_descriptions:
        continue
    
    UPCs_unified_decscriptions[UPC] = [UPC]
    
    for i in tqdm(range(len(product_descriptions[0])), position=1, leave=False):
        strings = [x[i].lower() for x in product_descriptions]
        
        # This assumes that number 3 and 8 benefit from the alternate aggregation method
        if i == 3 or i == 8:
            selected_word = find_min_average_distance_word(strings)
        else:
            selected_word = most_common_string(strings)
        
        UPCs_unified_decscriptions[UPC].append(selected_word)

# Save the results to a .csv
csv_path = args.csv_output if args.csv_output else f"{args.run_name}.csv"
data = [UPCs_unified_decscriptions[UPC] for UPC in UPCs]
header = ["UPC"] + header[1:]
save_csv(csv_path, data=data, header=header)