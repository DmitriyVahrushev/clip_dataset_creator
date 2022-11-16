from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from mean_average_precision import MetricBuilder
from sklearn.metrics import precision_score, recall_score


def compute_iou(pred, gt):
    """ Calculates IoU (Jaccard index) of two sets of bboxes:
            IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)
        Parameters:
            Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
            pred (np.array): predicted bboxes
            gt (np.array): ground truth bboxes
        Return value:
            iou (np.array): intersection over union
    """
    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.) * (box[:, 3] - box[:, 1] + 1.)

    _gt = np.tile(gt, (pred.shape[0], 1))
    _pred = np.repeat(pred, gt.shape[0], axis=0)
    ixmin = np.maximum(_gt[:, 0], _pred[:, 0])
    iymin = np.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = np.minimum(_gt[:, 2], _pred[:, 2])
    iymax = np.minimum(_gt[:, 3], _pred[:, 3])

    width = np.maximum(ixmax - ixmin + 1., 0)
    height = np.maximum(iymax - iymin + 1., 0)

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou

gt_df = pd.read_csv('misc/labels.csv')
preds_df = pd.read_csv('misc/preds_yolov5.csv')
gt_labels = gt_df.values
preds_labels = preds_df.values
num_samples = gt_labels.shape[0]

mAP_func = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
IoU_threshold = 0.5
preds_at_given_IoU = np.zeros(num_samples)
gt_at_given_IoU = np.ones(num_samples) # there is always one object in any given image in the dataset 

for i in range(num_samples):
    gt_bbox = np.expand_dims(gt_labels[i,:4], axis=0)
    pred_bbox = np.expand_dims(preds_labels[i,:4], axis=0)
    IoU_score = compute_iou(pred_bbox, gt_bbox)[0,0]
    print(i+2, IoU_score)
    label = 1 if IoU_score >= IoU_threshold else 0
    preds_at_given_IoU[i] = label
mAP_func.add(preds_labels, gt_labels)

precision_at_IoU = precision_score(gt_at_given_IoU, preds_at_given_IoU)
recall_at_IoU = recall_score(gt_at_given_IoU, preds_at_given_IoU)

print(f"Precision: {precision_at_IoU}, Recall: {recall_at_IoU} . IoU threshold: {IoU_threshold}")
print(f"VOC PASCAL mAP: {mAP_func.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
print(f"VOC PASCAL mAP in all points: {mAP_func.value(iou_thresholds=0.5)['mAP']}")
print(f"COCO mAP: {mAP_func.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
