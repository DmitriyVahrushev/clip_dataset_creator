from ast import arg
import os
import argparse
import time
import cv2
import pandas as pd
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
    parser.add_argument('--output_csv',type=str, help='path to output csv with bounding boxes coords',
                        default='bounding_boxes.csv')
    args = parser.parse_args()
    dataset_path = args.orig_dataset_folder
    output_path = args.output_path
    output_csv_path = args.output_csv
    query = args.query
    os.makedirs(output_path, exist_ok=True)

    if args.object_detector == "yolov5":
        object_detector = YoloV5ObjectDetector()
    elif args.object_detector == "class-agnostic":
        object_detector = ClassAgnosticObjectDetector()
    query_evaluator = QueryEvaluator()

    filenames = os.listdir(dataset_path)
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    preds_df = pd.DataFrame(columns=[
        'x_min', 'y_min', 'x_max', 'y_max', 'class_id', 'confidence'])
    for filename in filenames:
        if filename.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
            continue
        img = cv2.imread(f'{dataset_path}/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = object_detector.detect(f'{dataset_path}/{filename}')
        probs = query_evaluator.evaluate(img, query, bboxes)
        # find bbox with max confidence score
        res = np.argmax(probs) 
        res_confidence = np.max(probs)
        bb_coords = bboxes[res]
        # save results
        pred_row = [bb_coords[0],bb_coords[1],bb_coords[2], bb_coords[3], 
            0, res_confidence]
        preds_df.loc[len(preds_df)] = pred_row
        res_img = img[bb_coords[1]:bb_coords[3],bb_coords[0]:bb_coords[2]]
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{output_path}/{filename}', res_img)
    preds_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))