import torch

class ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    def detect(self, image_filename:str):
        results = self.model([image_filename])
        return results