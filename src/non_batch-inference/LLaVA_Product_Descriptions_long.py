# This script operates like llava.serve.cli
# You can use this script to run several question/image pairs through LLaVA
# Example usage:
#   python llava_custom_inference --model-path liuhaotian/llava-v1.5-13b --image-file products_imgs --load-4bit --composite_output composite_image.png
# The questions are hard coded in the class ImageQA
# Prompts include previous quetions and answers

# Standard library imports
import argparse
import csv
import datetime
import math
import os
import time
from io import BytesIO

# Related third party imports
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
#from line_profiler import LineProfiler

# Local application/library specific imports
from llava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images, 
    tokenizer_image_token, 
    get_model_name_from_path, 
    KeywordsStoppingCriteria
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import save_csv

# This gets appended to the first question
IMAGEQA_SYSTEM_MESSAGE = " Be accurate but brief. Only answer with text from the image, or if you don't know the answer, say 'unknown' or 'none'."

class Q_and_A:
    """Represents a question and its corresponding answer."""
    def __init__(self, question):
        self.question = question
        self.answer = None

class ImageQA:
    """Represents an image and a set of questions related to the image."""
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(self.image_path).convert('RGB')
        self.image_tensor = None
        self.questions = {"Section" : Q_and_A("Which supermarket section would have this product?"),
                          "Material" : Q_and_A("What material of packaging is it in?"),
                          "Shape" : Q_and_A("What shape of packaging is it in?"),
                          #"Facing" : Q_and_A("Which side of this product are you looking at?"),
                          "Color_Main" : Q_and_A("What is the primary color?"),
                          "Color_Secondary" : Q_and_A("What is the secondary color?"),
                          #"Text" : Q_and_A("Yes or no: Is there any readable text?"),
                          "Brand" : Q_and_A("What brand is it?"),
                          "Product" : Q_and_A("What base type of product is it?"),
                          "Type" : Q_and_A("What flavor, type, or variant is it?"),
                          "Size" : Q_and_A("Estimate the size, including units."),
                          "Other text" : Q_and_A("Besides the text you already mentioned, what other text is there, if any?"),
                          "Distinct feature" : Q_and_A("What distinct visual features are there, if any?"),}
                          #"Nutrition" : Q_and_A("What nutritional information can you read?"),
            
        # Append a message to the first question to encourage short answers and accurate answers
        self.questions[next(iter(self.questions))].question += IMAGEQA_SYSTEM_MESSAGE

    @staticmethod
    def process_images(imageQAs, image_processor, model):
        """Processes a list of images for inference."""
        images_list = [imageQA.image for imageQA in imageQAs]
        image_tensors = process_images(images_list, image_processor, model.config)
        image_tensors = [image.to(model.device, dtype=torch.float16) for image in image_tensors]
            
        for imageQA, image_tensor in zip(imageQAs, image_tensors):
            imageQA.image_tensor = image_tensor.unsqueeze(0)

def load_images(image_file):
    """Load and process images from a given file or directory."""
    image_paths = []
    if os.path.isfile(image_file):
        image_paths = [image_file]
    elif os.path.isdir(image_file):
        image_paths = [os.path.join(image_file, f) for f in os.listdir(image_file)]
    image_paths = [f for f in image_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_paths:
        print(f"[WARNING] No image found at {image_file}")
        return []
    
    return image_paths

def load_model(args):
    """Load an model from the given arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_known_args(args)[0]
    
    global tokenizer, model, image_processor, conv_mode

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        conv_mode = args.conv_mode
    
    return tokenizer, model, image_processor, conv_mode
    #roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv_templates[conv_mode].roles

#@profile
def inference(args, tokenizer, model, image_processor, conv_mode):
    #print(args)
    """Run inference using the previously loaded model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0) # Original default is 0.2
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true", help="Show the dialogue as the model sees it")
    
    # Choose at least one output method
    #  --verbose           CLI output
    #  --composite_output  Create a composite image
    #  --csv_output        Save to a CSV
    parser.add_argument("--verbose", action="store_true", help="Set true to output the dialogue as it is generated")
    parser.add_argument("--composite_output", type=str, help="The path of the composite output image", default=None)
    parser.add_argument("--csv_output", type=str, help="The path of the output csv", default=None)
    args = parser.parse_known_args(args)[0]

    # Create imageQA objects
    image_paths = load_images(args.image_file)
    if not image_paths:
        return
    imageQAs = [ImageQA(image_path) for image_path in image_paths]
    ImageQA.process_images(imageQAs, image_processor, model)
    #print(f"num imageQAs: {len(imageQAs)}")
    
    image_load_time = time.time()
    
    if args.verbose:
        print(f"Starting inference")
    
    num_questions = sum([len(imageQA.questions) for imageQA in imageQAs])
    question_counter = 0
    
    for imageQA in imageQAs:
        conv = conv_templates[conv_mode].copy() # Reset the conversation
        
        first_question = True # True if the first question has not been asked yet
        
        # Iterate over the keys to the question dictionary
        for question_index, question_key in enumerate(imageQA.questions):
            # Get the question
            Q_and_A = imageQA.questions[question_key]
            
            question_counter += 1
            progress = f"{question_counter/num_questions:.0%}"
            image_name = os.path.basename(imageQA.image_path)
            progress_text = f"{progress} {image_name} Asking question {question_index+1} of {len(imageQA.questions)}".ljust(60)
            print(progress_text, end = "\r")
            
            question_text = Q_and_A.question
            
            # Process the first message in a special way
            if first_question:
                if model.config.mm_use_im_start_end:
                    question_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_text
                else:
                    question_text = DEFAULT_IMAGE_TOKEN  + question_text
                first_question = False
            
            # Add update the conversation
            conv.append_message(conv.roles[0], question_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Prepare for inference
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            # Run inference
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=imageQA.image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            
            # Decode the output and update the answer to the question
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            Q_and_A.answer = outputs[:-4].lower()
        
        # Print the dialogue as it is actually seen by the model.
        if args.debug:
            print(f"\nFinal dialogue: {conv.get_prompt()}\n")
        
        # Print the answers to the questions, formatted nicely
        if args.verbose:
            print()
            for question_key, Q_and_A in imageQA.questions.items():
                print(f"{question_key}\t{Q_and_A.answer}")
    
    # A helper function for formatting elapsed time
    def get_elapsed_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_seconds = datetime.timedelta(seconds=elapsed_time).total_seconds()
        return elapsed_seconds
    
    # Time tracking
    end_time = time.time()
    total_time = get_elapsed_time(image_load_time, end_time)
    if args.verbose:
        print(f"\nTotal inference time: {total_time:.2f} seconds")
        if len(imageQAs) > 1:
            time_per_image = total_time / len(imageQAs)
            print(f"Inference time per image: {time_per_image:.2f} seconds")
        time_per_question = total_time / sum([len(x.questions) for x in imageQAs])
        print(f"Inference time per question: {time_per_question:.2f} seconds")
    
    # Save as composite image
    if args.composite_output:
        composite_image = create_composite_image(imageQAs)
        #composite_image.show()  # Display the image
        composite_image.save(args.composite_output)  # Save the image to a file
    
    # Save as CSV
    if args.csv_output:
        save_results_to_csv(args.csv_output, imageQAs)
    
    # Return the results
    return imageQAs

def save_results_to_csv(csv_path, imageQAs):
    header = []
    header.append("image")
    for question in imageQAs[0].questions:
        header.append(question)

    data = []
    for imageQA in imageQAs:
        answers = [os.path.basename(imageQA.image_path)]
        for question_key, Q_and_A in imageQA.questions.items():
            answers.append(Q_and_A.answer)
        data.append(answers)

    save_csv(csv_path, data=data, header=header)

def create_composite_image(imageQAs):
    title_padding = 15
    title = IMAGEQA_SYSTEM_MESSAGE
    
    images_per_row = math.ceil(math.sqrt(len(imageQAs)))
    image_horizontal_padding = 5
    image_vertical_padding = 0
    
    # Determine the target height based on the average image
    target_width = sum(imageQA.image.width for imageQA in imageQAs)//len(imageQAs)

    # Resize images to have the same height while maintaining aspect ratio
    for imageQA in imageQAs:
        #print(f"original size: {imageQA.image.width}, {imageQA.image.height}")
        aspect_ratio = imageQA.image.height / imageQA.image.width
        new_height = int(target_width * aspect_ratio)
        resized_image = imageQA.image.resize((target_width, new_height))
        imageQA.image = resized_image  # Update the imageQA with the resized image
        
        #print(f"new size: {imageQA.image.width}, {imageQA.image.height}")

    # Calculate the width and height of each row
    max_row_width = 0
    for x in range(0, len(imageQAs), images_per_row):
        row_width = sum([image.width for image in [imageQA.image for imageQA in imageQAs[x:x+images_per_row]]])
        if row_width > max_row_width:
            max_row_width = row_width
    max_image_height = max([imageQA.image.height for imageQA in imageQAs])
    max_text_height = max([15 * (len(imageQA.questions) + 1) for imageQA in imageQAs])
    
    total_width = max_row_width + (image_horizontal_padding * (images_per_row - 1))
    total_rows = math.ceil(len(imageQAs) / images_per_row)
    total_height = (max_image_height + max_text_height) * total_rows + (image_vertical_padding * (total_rows - 1))

    # Create a blank composite image with white background
    composite_image = Image.new('RGB', (total_width, total_height + title_padding), 'white')
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(composite_image)
    
    draw.text((0,0), title, fill='black', font=font)

    # Initialize variables to keep track of the current x and y positions
    x_position = 0
    y_position = 0

    # Iterate through each ImageQA object and their answers
    for idx, imageQA in enumerate(imageQAs):
        # If we have reached the maximum number of images per row, reset x_position and update y_position
        if idx % images_per_row == 0 and idx != 0:
            x_position = 0
            y_position += max_image_height + max_text_height + image_vertical_padding

        # Paste the image onto the composite image
        composite_image.paste(imageQA.image, (x_position, y_position + title_padding))

        # Draw the answer text underneath the image
        text_y_offset = imageQA.image.height
        draw.text((x_position, y_position + text_y_offset + title_padding), os.path.basename(imageQA.image_path), fill='black', font=font)
        text_y_offset += 15
        for question_key, Q_and_A in imageQA.questions.items():
            draw.text((x_position, y_position + text_y_offset + title_padding), f"{question_key}:".ljust(12) + str(Q_and_A.answer), fill='black', font=font)
            text_y_offset += 15

        # Update the x-position for the next image
        x_position += imageQA.image.width + image_horizontal_padding

    return composite_image
    
if __name__ == "__main__":
    inference_parts = load_model(args=None)
    inference(None, *inference_parts)