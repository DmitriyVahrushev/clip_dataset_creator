import torch

import tensorflow as tf
import tensorflow_hub as hub

class YoloV5ObjectDetector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    def detect(self, image_filename:str):
        results = self.model([image_filename])
        return results.xyxy[0].cpu().numpy().astype('int')


def _load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

class ClassAgnosticObjectDetector:
    def __init__(self) -> None:
        module_handle = "https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1"
        self.detector = hub.load(module_handle).signatures['default']

    def detect(self, image_filename:str):
        img = _load_img(image_filename)
        img_width, img_length = img.shape[1], img.shape[0]
        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = self.detector(converted_img)
        detection_boxes = result['detection_boxes'][0].numpy()
        detection_boxes[:,0] = img_width*detection_boxes[:,0]
        detection_boxes[:,1] = img_length*detection_boxes[:,1]
        detection_boxes[:,2] = img_width*detection_boxes[:,2]
        detection_boxes[:,3] = img_length*detection_boxes[:,3]
        return detection_boxes.astype('int')
