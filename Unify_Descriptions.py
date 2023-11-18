# This script takes one argument:
#  the path to fv_albt_test_11152023
# This script unifies the product descriptions contained in that folders

import argparse
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", help="The path to fv_ablt_test")
args = parser.parse_args()

# Read UPCs
UPCs = os.listdir(args.reference_images)
UPCs = [UPC for UPC in UPCs if UPC[0] != '.']
UPCs = dict.fromkeys(UPCs, None)

def read_csv(file, skip_header):
    """Returns the contents of a .csv file"""
    data = []
    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if skip_header:
                skip_header = False
                continue
            data.append(row)
    return data

# Read the .csv files
for UPC in UPCs:
    csv_path = os.path.join(args.reference_images, UPC, f"{UPC}.csv")
    if os.path.isfile(csv_path):
        UPCs[UPC] = [row[1:] for row in read_csv(csv_path, skip_header=True)]

# De-collate the decsriptions
for UPC, product_descriptions in UPCs.items():
    for i in range(len(product_descriptions[0])):
        print([x[i] for x in product_descriptions])
