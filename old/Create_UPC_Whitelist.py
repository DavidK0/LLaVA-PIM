# This script is used when you want to distribute a task to multiple machines.
# It will figure out what UPCs have not yet been processed, then split them into two lists.
# You can pass one list off to one machine as a whitelist, and the other of to another machine.

import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store")
parser.add_argument("product_descriptions", type=str, action="store")
parser.add_argument("output_name", type=str, action="store", help="The name of the output files")
args = parser.parse_args()

reference_upcs = os.listdir(args.reference_images)
processed_upcs = [file.split(".")[0] for file in os.listdir(args.product_descriptions)]

unprocessed_upcs = list(set(reference_upcs).difference(set(processed_upcs)))

first_half = unprocessed_upcs[:len(unprocessed_upcs)//2]
second_half = unprocessed_upcs[len(unprocessed_upcs)//2:]

# Save the first half
with open(f"{args.output_name}_1.txt", "w") as file:
    for upc in first_half:
        file.write(f"{upc}\n")
        
# Save the second half
with open(f"{args.output_name}_2.txt", "w") as file:
    for upc in second_half:
        file.write(f"{upc}\n")