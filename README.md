# clip_dataset_creator
![Example image](misc/example_img.jpg)
Command line tool for labeled dataset creation with CLIP

## Usage example
```
pip install -r requirements.txt
python main.py --orig_dataset_folder images --query "red car" --output_folder dataset --object_detector yolov5
```
## Metrics
To estimate the quality of project I created small dataset with car images. mAP metrics below calculated for query "red car" on this dataset. The dataset can be found in 'images' folder, labels are in 'misc' folder. <br>
Presion and recall values were calculated at IoU threshold of 0.5. <br>
Note that the size of the dataset is very small.

| Model name | VOC PASCAL mAP | VOC PASCAL mAP in all points | COCO mAP | Presicion | Recall
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
YOLOV5+CLIP | 1.0 | 1.0 | 0.74803626537323 | 1.0 | 1.0
Class agnostic object detector+CLIP | 0.0844 | 0.07755102217197418 | 0.010289957746863365 | 1.0 | 0.09524

## Performance
~0.32s per image on Nvidia RTX3060. Calculated using the dataset discribed above.

## System requirements
* Nvidia GPU with minimum 4GB VRAM 