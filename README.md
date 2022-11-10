# clip_dataset_creator
![Example image](misc/example_img.jpg)
Command line tool for labeled dataset creation with CLIP

## Usage example
```
pip install -r requirements.txt
python main.py --orig_dataset_folder images --query "red car" --output_folder dataset --object_detector yolov5
```
## Metrics
To estimate the quality of project I created small dataset with car images. mAP metrics below calculated for query "red car" on this dataset. The dataset can be found in 'images' folder, labels are in 'misc' folder.

YOLOV5+CLIP <br>
VOC PASCAL mAP: 0.4848484992980957 <br>
VOC PASCAL mAP in all points: 0.46666666865348816 <br>
COCO mAP: 0.34772181510925293 <br>

Class agnostic object detector (performs significantly worse): <br>
VOC PASCAL mAP: 0.022727273404598236 <br>
VOC PASCAL mAP in all points: 0.011904762126505375 <br>
COCO mAP: 0.002475247485563159

## Performance
~0.32s per image on Nvidia RTX3060. Calculated using the dataset discribed above.

## System requirements
* Nvidia GPU with minimum 4GB VRAM 