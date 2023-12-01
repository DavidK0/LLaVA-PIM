import argparse
from utils import read_csv

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=str, action="store", help="Path to a .csv file")
args = parser.parse_args()

header, data = read_csv(args.csv_file, has_header=True)

label_ids = {}
for row in data:
    label_id = row[header.index("label_id")]
    if label_id not in label_ids:
        label_ids[label_id] = []
    label_ids[label_id].append(row)

user_code_in_candidates_total = 0
for label_id, data in label_ids.items():
    code_from_user = data[0][header.index("code_from_user")] # This assumes all rows have the same code_from_user
    user_code_in_candidates = code_from_user in [row[header.index("candidate_upc")] for row in data]
    if user_code_in_candidates:
        user_code_in_candidates_total += 1
rate = user_code_in_candidates_total / len(label_ids)
print(f"{rate:.1%} of labels have code_from_user somewhere in the candidate_upcs")
print(f"{1-rate:.1%} of labels are missing code_from_user in the candidate_upcs")