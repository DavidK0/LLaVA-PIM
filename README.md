# LLaVA-PIM
In this repo I use the vision-language model LLaVA to extract product information and classify products based on their images. The goal is to start with a cropped image of some product sitting on a shelf, and extract as much product information from that image as possible (eg. brand, size, nutritional information, etc.), then compare these product descriptions together to classify them.

# How to use

There are two ways to use this: with batch-inference (faster but limited to llava-7b) or without batch-inference (slower but can use llava-13b).

## Product descriptions with batch-inference
1. Follow the instruction for setting up [the dev branch of LLaVA](https://github.com/haotian-liu/LLaVA/tree/develop)
2. Copy two files to LLaVA: `cp LLaVA-PIM/src/batch-inference/Custom_Batch_Inference.py LLaVA/` and `cp LLaVA-PIM/src/batch-inference/Make_Questions_jsonl_long.py LLaVA/`
3. `cd LLaVA`
4. Get the product descriptions with `python Make_Questions_jsonl_long.py input_dir output_dir`. The input_dir should hold one folder for each product, and images of those products go in their respective folders.
5. Unify the product descriptions into a single description per product with `python LLaVA-PIM/src/batch-inference/Unify_Batch_Descriptions.py input_dir output_file` where input_dir is the output_dir from the previous step.

## Product descriptions without batched-inference
1. Follow the instruction for setting up [the main branch of LLaVA](https://github.com/haotian-liu/LLaVA/)
2. `cd LLaVA-PIM`
3. Get the product descriptions with `python src/non_batch-inference/Process_Images.py input_dir output_dir'`. The input_dir should hold one folder for each product, and images of those products go in their respective folders.
4. Unify the product descriptions into a single description per product with `python src/non_batch-inference/Unify_Descriptions.py input_dir output_file` where input_dir is the output_dir from the previous step.

## Product descriptions for a single image or product
This is not intended to be used for later classifying products.
1. Follow the instruction for setting up [the main branch of LLaVA](https://github.com/haotian-liu/LLaVA/)
2. `cd LLaVA-PIM`
3. Get the product descriptions with `python src/LLaVA_Product_Descriptions_long.py --model-path liuhaotian/llava-v1.5-13b --image-file input_file_or_dir --load-8bit`.  Pick one of the three output methods (`--verobse`, `--composite-output`, or `--csv-output`) The input should be a single images or folder of images.

## Classifying from product descriptions
Regardless of if you used batch-inference or not, the output_file will hold all the product descriptions in .jsonl format. Use `python LLaVA-PIM/src/Classify_Products.py reference_descriptions input_descriptions pl_info output` to get the description of a new image or images and compare them against the previously generated and unified descriptions. reference_descriptions and input_descriptions are the product descriptions in .jsonl files generated on previous steps.

## Long versus Short
Both the batched and non-batched versions have long and short sub-version. 'Long' refers to the fact that the prompts will always include answers to previously asked questions. 'Short' refers to the fact the those prompts will be shortened to just the current question. By default the long version is used, and there is no argument for changing it.

# Evaluation
This system picks the correct pl candidate 75% of the time.

# Improvements
In this section I will consider possible improvements to the system and other things to consider.
1. Having a 'confidence score' would allow the system to avoid using product descriptions for poorly faced items as reference descriptions.
2. If the model were better at determining which face it is looking at (it is currently bad at this), then we could model/describe the faces seperately
3. Reduced STR hallucinations (by using eg. CogVLM) would help a lot