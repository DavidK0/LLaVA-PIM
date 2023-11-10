# This script operates like llava.serve.cli
# You can use this script to run several question/image pairs through LLaVA
# Example usage:
#   python llava_custom_inference --model-path liuhaotian/llava-v1.5-13b --image-file products_imgs --load-4bit

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
#from transformers import TextStreamer

import time
import datetime
import os

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

# A helper function for formatting elapsed time
def get_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_seconds = datetime.timedelta(seconds=elapsed_time).total_seconds()
    return f"{elapsed_seconds:.2f} seconds"

# This class represents one question and one answer
class Q_and_A:
    # Example question: "What brand is this?"
    # Corresponding short_form: "Brand"
    def __init__(self, question, short_form):
        self.question = question
        self.short_form = short_form
        self.answer = None

# Create the list of questions
Q_and_As = []
Q_and_As.append(Q_and_A("What type of product is this?", "Type"))
Q_and_As.append(Q_and_A("What brand is this?", "Brand"))
Q_and_As.append(Q_and_A("What flavor is it?", "Flavor"))
Q_and_As.append(Q_and_A("What style package is it in?", "Packaging"))
#Q_and_As.append(Q_and_A("Is any of this text nutritional information? If so what does it say?", "Nutritional check2"))
Q_and_As.append(Q_and_A("Is there any nutritional text you can read? If so, what does it say?", "Nutritional check"))
#Q_and_As.append(Q_and_A("What nutritional information can you read", "Nutritional"))
#Q_and_As.append(Q_and_A("Is there any readable text that tell you the size of this?", "Size check"))
#Q_and_As.append(Q_and_A("What size information can you read", "Size"))

# Append a message to the first question to encourage short answers
#Q_and_As[0].question += "Don't use full sentences."

def main(args):
    start_time = time.time() # Time tracking
    
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

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
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    # Load the image or images
    images = []
    if os.path.isfile(args.image_file):  # Check if the argument is a file
        images.append(Image.open(args.image_file))
    elif os.path.isdir(args.image_file):  # Check if the argument is a directory
        for filename in os.listdir(args.image_file):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(args.image_file, filename)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
    
    # Similar operation in model_worker.py
    print(images)
    image_tensors = process_images(images, image_processor, model.config)
    if type(image_tensors) is list:
        image_tensors = [image.to(model.device, dtype=torch.float16) for image in image_tensors]
    else:
        image_tensors = image_tensors.to(model.device, dtype=torch.float16)
    
    image_load_time = time.time()
    print(f"Starting inference")
    
    for image_tensor in image_tensors:
        image_tensor = image_tensor.unsqueeze(0)
        
        conv = conv_templates[args.conv_mode].copy() # Reset the conversation
        first_question = True # True if the first question has not been asked yet
        
        print()
        for question_index, Q_and_A in enumerate(Q_and_As):
            print(f"Asking question {question_index+1} of {len(Q_and_As)}", end = "\r")
            inp = Q_and_A.question

            if first_question:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                first_question = False
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    #streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            Q_and_A.answer = outputs[:-4]

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    
        #final_prompt = conv.get_prompt()
        #print(final_prompt)
        print()
        for Q_and_A in Q_and_As:
            print(f"{Q_and_A.short_form}: {Q_and_A.answer}")
    
    end_time = time.time() # Time tracking
    print(f"Total inference time: {get_elapsed_time(image_load_time, end_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0) # Original default is 0.2
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)