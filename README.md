# LLaVA-PIM
In this repo I use the vision-language model LLaVA to extract product information from shelf images. The goal is to start with a cropped image of some product sitting on a shelf, and extract as much product information from that image as possible (eg. brand, size, nutritional information, etc.).

# How to use

1. Follow the instruction for setting up [LLaVA](https://github.com/haotian-liu/LLaVA/)
2. Clond this repo
3. Run something like `python llava_custom_inference --model-path liuhaotian/llava-v1.5-13b --image-file imgs_folder --load-4bit`

The questions are hard-coded.
