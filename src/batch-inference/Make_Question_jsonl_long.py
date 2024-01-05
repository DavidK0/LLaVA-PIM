# This is the main script for processing images into product descriptions using batch inference.
# It takes two arguments:
#   reference_images: this is a folder a folder of images of products
#   output_dir: this folder will be created and the resulting product descriptions will be placed there

import os
import sys
import json
import shutil
import argparse

import Custom_Batch_Inference

parser = argparse.ArgumentParser()
parser.add_argument("reference_images", type=str, action="store", default=None, help="This is a folder a folder of images of products")
parser.add_argument("output_dir", type=str, action="store", default=None, help="This folder will be created and the resulting product descriptions will be placed there")
args = parser.parse_args()


# Load image paths
image_paths = []
for root, dirs, files in os.walk(args.reference_images):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_paths.append(os.path.join(root, file))
if not image_paths:
    raise ValueError(f"No images found in {args.reference_images}")

# Create the output dir
if not os.path.exists(args.output_dir):
    print(f"Creating {args.output_dir}")
    os.mkdir(args.output_dir)

# Create flattened image folder
image_folder = os.path.join(args.output_dir, "images")
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Flatten images
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
for root, dirs, files in os.walk(args.reference_images):
    for file in files:
        if file.lower().endswith(tuple(image_extensions)):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(image_folder, file)
            shutil.copy2(source_path, destination_path)

system_message = "Answer accurately but use only a few words. Only answer with text from the image, or if you don't know the answer, say 'unknown' or 'none'."
questions = ["Tell me, which supermarket section would have this product?",
             "Tell me, what material of packaging is this product in?",
             "Tell me, what shape of packaging is this product in?",
             "Tell me, what is the primary color of this product?",
             f"Tell me, what is the secondary color of this product?",
             "Tell me, what brand is this product?",
             "Tell me, what is the primary category of this product?",
             "Tell me, within its primary category, what flavor, type, or variant is this product?",
             "Estimate the size of this product, including units.",
             "Tell me, what is the most visually distinct feature about this product?"]


for question_index, question in enumerate(questions):
    question_file = f"{args.output_dir}_question_{question_index}.jsonl"
    question_file_path = os.path.join(args.output_dir, question_file)
    
    
    answer_file = f"{args.output_dir}_answer_{question_index}.jsonl"
    answer_file_path = os.path.join(args.output_dir, answer_file)
    
    if not os.path.isfile(question_file_path):
        # Get previous question/answers
        previous_convos = []
        if question_index > 0:
            previous_answer_file = f"{args.output_dir}_answer_{question_index - 1}.jsonl"
            previous_answer_file_path = os.path.join(args.output_dir, previous_answer_file)
            
            conversations = [json.loads(line) for line in open(previous_answer_file_path)]
            for conversation in conversations:
                question_ = conversation["prompt"]
                answer = conversation["text"]
                previous_convos.append(question_ + [answer])
                
        
        # Create the next questions file
        with open(question_file_path, "w") as file:
            question_counter = 0
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                
                if previous_convos:
                    text = previous_convos[question_counter] + [question]
                else:
                    text = [question]
                
                file.write(json.dumps({"question_id": question_counter,
                                       "image": image_name,
                                       "system_message": system_message,
                                       "text": text,
                                       "category": "conv"}) + "\n",)
                question_counter += 1
        
        # Execute batch inference with the questions file
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
        parser.add_argument("--model-base", type=str, default=None)
        parser.add_argument("--image-folder", type=str, default="")
        parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
        parser.add_argument("--answers-file", type=str, default="answer.jsonl")
        parser.add_argument("--conv-mode", type=str, default="llava_v1")
        parser.add_argument("--num-chunks", type=int, default=1)
        parser.add_argument("--chunk-idx", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--max_new_tokens", type=int, default=128)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=4)
        print(image_folder, question_file_path)
        batch_inference_args = ["--model-path", "liuhaotian/llava-v1.5-7b", "--image-folder", image_folder, "--question-file", question_file_path, "--answers-file", answer_file_path, "--temperature", "0", "--batch_size", "5"]
        batch_inference_args = parser.parse_args(batch_inference_args)
        
        Custom_Batch_Inference.eval_model(batch_inference_args)