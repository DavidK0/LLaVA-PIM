import os
import argparse
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a folder and calculate the average number of files in subfolders.")
    parser.add_argument('folder', type=str, help='Path to the folder to process')
    args = parser.parse_args()

    # Process the specified folder
    process_folder(args.folder)

def process_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory or does not exist.")
        return

    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    total_files = 0
    subfolder_count = 0

    #print(f"Contents of {folder_path}:")

    # Traverse each item in the directory
    for entry in tqdm(os.scandir(folder_path)):
        #print(f" - {entry.name}")
        if entry.is_dir():
            file_count = len([f for f in os.scandir(entry.path) if f.is_file()])
            #print(f"   Subfolder '{entry.name}' contains {file_count} files.")
            total_files += file_count
            subfolder_count += 1

    # Calculate average number of files in each subfolder
    if subfolder_count > 0:
        average_files = total_files / subfolder_count
        print(f"Average number of files in each subfolder: {average_files:.2f}")
    else:
        print("No subfolders found.")

if __name__ == '__main__':
    main()