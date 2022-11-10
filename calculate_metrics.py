from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from mean_average_precision import MetricBuilder


gt_df = pd.read_csv('misc/labels.csv')
gt_labels = gt_df.values
preds_df = pd.read_csv('misc/preds_class_agnostic_obj_det.csv')
preds_labels = preds_df.values

metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)

for i in range(preds_labels.shape[0]):
    if i<=2: # quickfix for the error in label csv markup
        gt_label = np.expand_dims(gt_labels[i], axis=0) # between 3 and 4
        pred_label = np.expand_dims(preds_labels[i], axis=0)
        metric_fn.add(pred_label, gt_label)
    else:
        gt_labels[i,1], gt_labels[i,3] = gt_labels[i,3], gt_labels[i,1]
        gt_label = np.expand_dims(gt_labels[i], axis=0) # between 3 and 4
        pred_label = np.expand_dims(preds_labels[i], axis=0)
        metric_fn.add(pred_label, gt_label)

# compute PASCAL VOC metric
print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
# compute metric COCO metric
print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
