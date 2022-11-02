# clip_dataset_creator
![Example image](misc/example_img.jpg)
Command line tool for labeled dataset creation with CLIP

## Usage example
```
pip install -r requirements.txt
python main.py --orig_dataset_folder images --query "red car" --output_folder dataset
```
## Metrics
TO DO

## Performance
~0.12s per image on Nvidia Tesla T4

## System requirements
* Nvidia GPU with minimum 8GB VRAM 