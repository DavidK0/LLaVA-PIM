# This script is for comparing the 7b llava model to the 13b model.
# It takes two arguments:
#   a .csv containing product descriptions for several images
#   another .csv, the same length as the first and with the same images
# This script will output any differences.

import argparse
import sys
import csv
import os

from utils import normalized_edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("prod_descriptions1")
parser.add_argument("prod_descriptions2")
args = parser.parse_args()

product_descriptions1 = {}
product_descriptions2 = {}

# Get descriptions from the first file
with open(args.prod_descriptions1) as file:
    csv_reader = csv.reader(file)
    for row in list(csv_reader)[1:]:
        image_name = row[0]
        product_description = row[1:]
        product_descriptions1[image_name] = product_description

# Get descriptions from the second file
with open(args.prod_descriptions2) as file:
    csv_reader = csv.reader(file)
    for row in list(csv_reader)[1:]:
        image_name = row[0]
        product_description = row[1:]
        product_descriptions2[image_name] = product_description

# Check the the files have the same images
for image_name in product_descriptions1:
    if image_name not in product_descriptions2:
        raise Exception(f"{image_name} is missing from {args.prod_descriptions2}")
for image_name in product_descriptions2:
    if image_name not in product_descriptions1:
        raise Exception(f"{image_name} is missing from {args.prod_descriptions1}")

# Find differences
total_average_distance = 0
for image_name in product_descriptions1:
    description1 = product_descriptions1[image_name]
    description2 = product_descriptions2[image_name]
    
    # Check that the product descriptions are the same length
    if not len(description1) == len(description2):
        raise Exception(f"{image_name} has mis-matched product description lengths ({len(description1)} and {len(description2)})")
    
    # Find average NED between the two descriptions
    description_distance = 0
    for element1, element2 in zip(description1, description2):
        description_distance += normalized_edit_distance(element1, element2)
    average_element_distance = description_distance / len(description1)
    
    total_average_distance += average_element_distance
    
    DIFFERENCE_THRESHOLD = .01 # Only print descriptions if they are at least this far apart
    if average_element_distance >= DIFFERENCE_THRESHOLD:
        print(f"{image_name} {', '.join(description1)}")
        print(f"{str(round(average_element_distance,2)).ljust(len(image_name))} {', '.join(description2)}")
print(f"\nAverage NED: {total_average_distance/len(product_descriptions1):.2f}")
#print((product_descriptions1))