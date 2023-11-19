# LLaVA-PIM
In this repo I use the vision-language model LLaVA to extract product information and classify products based on their images. The goal is to start with a cropped image of some product sitting on a shelf, and extract as much product information from that image as possible (eg. brand, size, nutritional information, etc.), then compare these product descriptions together to classify them.

# How to use for extracting product information
1. Follow the instruction for setting up [LLaVA](https://github.com/haotian-liu/LLaVA/)
2. Clone this repo
3. To run inference on a single image or a directory, run something like `python LLaVA_Product_Descriptions.py --model-path liuhaotian/llava-v1.5-13b --image-file imgs_folder --load-8bit`. Pick one of the three output methods (`--verobse`, `--composite-output`, or `--csv-output`)

# How to use for product classification
1. Run something like `python Process_Reference_Images.py reference_folder --model-path liuhaotian/llava-v1.5-13b --run-name llava-13b` where 'reference_folder' is a folder of folders of images. The internal folders represent different products with different UPCs. A .csv files will be created inside each internal folder holding all the description for each of the images.
2. Use `Unify_Descriptions.py` to get a single product description for each UPC.
3. Use `Classify_Products.py` to get the description of a new image or images and compare them against the previously generated and unified descriptions.
