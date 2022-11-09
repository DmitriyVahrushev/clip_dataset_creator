import os
import argparse
import time
import cv2
import numpy as np
from object_detection import YoloV5ObjectDetector, ClassAgnosticObjectDetector
from query_evaluation import QueryEvaluator


def main():
    parser = argparse.ArgumentParser(description='Process inputs.')
    parser.add_argument('--orig_dataset_folder', type=str,
                        help='path to original dataset folder with images')
    parser.add_argument('--output_path', type=str,
                        help='path to output folder', default='output')
    parser.add_argument('--query', type=str,
                        help='text query for CLIP')
    parser.add_argument('--object_detector', type=str,
                        help='object detector model. Possible values: "yolov5", "class-agnostic"',
                        default='class-agnostic')
    args = parser.parse_args()
    dataset_path = args.orig_dataset_folder
    output_path = args.output_path
    query = args.query
    os.makedirs(output_path, exist_ok=True)

    if args.object_detector == "yolov5":
        object_detector = YoloV5ObjectDetector()
    elif args.object_detector == "class-agnostic":
        object_detector = ClassAgnosticObjectDetector()
    query_evaluator = QueryEvaluator()

    filenames = os.listdir(dataset_path)
    for filename in filenames:
        if filename.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
            continue
        img = cv2.imread(f'{dataset_path}/{filename}')
        bboxes = object_detector.detect(f'{dataset_path}/{filename}')
        bboxes = 515*bboxes
        probs = query_evaluator.evaluate(img, query, bboxes)
        print(probs)
        res = np.argmax(probs)
        bb_coords = bboxes[res].cpu().numpy().astype('int')
        print(bb_coords[1],bb_coords[3],bb_coords[0],bb_coords[2])
        res_img = img[bb_coords[1]:bb_coords[3],bb_coords[0]:bb_coords[2]]
        cv2.imwrite(f'{output_path}/{filename}', res_img)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))