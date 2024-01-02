# This script is for checking the how similar the product descriptons are that come from a
#   single UPC. The ideas is that that if the product descriptions for all the reference images
#   are similar, that suggests that the VLM is doing a good job at getting the 'right' information.
#   On the other hand if product descriptions vary a lot, they might not be reliable.

import argparse
import sys
import csv
import os

from utils import normalized_edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("reference_images1")
parser.add_argument("reference_images2")
args = parser.parse_args()

csv_files1 = []
for root, dirs, files in os.walk(args.reference_images1):
    csv_files1.extend(files)
csv_files1 = [f for f in csv_files1 if ".csv" in f and "checkpoint" not in f]

csv_files2 = []
for root, dirs, files in os.walk(args.reference_images2):
    csv_files2.extend(files)
csv_files2 = [f for f in csv_files2 if ".csv" in f and "checkpoint" not in f]

csv_files = list(set(csv_files1).intersection(set(csv_files2)))

skipped_upcs = 0
total_total = 0
total_pairs = 0
for csv_file in csv_files:
    csv_path = os.path.join(args.reference_images1,csv_file)
    if not os.path.exists(csv_path):
        continue
    
    with open(csv_path) as file:
        csv_reader = csv.reader(file)
        product_descriptions = [row[1:] for row in csv_reader][1:]
    
    if len(product_descriptions) < 2:
        skipped_upcs += 1
        continue
    
    for i in range(len(product_descriptions[0])):
        total = 0
        for x in range(len(product_descriptions)):
            for y in range(len(product_descriptions)):
                if x == y:
                    continue
                
                ned = normalized_edit_distance(product_descriptions[x][i], product_descriptions[y][i])
                total += ned
                print(ned, product_descriptions[x][i], product_descriptions[y][i])
        pairs = len(product_descriptions) * (len(product_descriptions) - 1)

        average_ned = total / pairs
        
        #print("\t",average_ned, pairs, total, len(product_descriptions))
        
        total_total += total
        total_pairs += pairs

print(f"Total UPCs: {len(csv_files)}")
print(f"UPCs skipped: {skipped_upcs} ({skipped_upcs/len(csv_files):.1%})")
print(f"Average NED: {total_total/total_pairs:.2f}")