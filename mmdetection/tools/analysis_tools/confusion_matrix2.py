"""
python mmdetection/tools/analysis_tools/confusion_matrix2.py --submission-csv work_dirs/faster_rcnn_r50_fpn_2x_coco/latest/submission.csv
--> work_dirs/faster_rcnn_r50_fpn_2x_coco/latest/confusion_matrix.png 경로로 이미지가 생성됨
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from pycocotools.coco import COCO
from tqdm import tqdm

GT_JSON_PATH = "/opt/ml/dataset/train_all.json"
CLASSES = [
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


def box_iou_calc(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])

    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    
    # <https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py>
    """
    # box = 4xn
    box_area = lambda box: (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


@dataclass
class ConfusionMatrix:
    classes: list = CLASSES
    num_classes: int = len(classes)
    CONF_THRESHOLD: float = 0.3
    IOU_THRESHOLD: float = 0.5
    matrix: np.ndarray = np.zeros((num_classes + 1, num_classes + 1))

    def plot(self, filepath: Union[str, Path]):
        array = self.matrix / (
            self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1e-6
        )  # normalize
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
        # apply names to ticklabels
        labels = (0 < len(self.classes) < 99) and len(self.classes) == self.num_classes
        sn.heatmap(
            array,
            annot=self.num_classes < 30,
            annot_kws={"size": 8},
            cmap="Blues",
            fmt=".2f",
            square=True,
            xticklabels=self.classes + ["background FP"] if labels else "auto",
            yticklabels=self.classes + ["background FN"] if labels else "auto",
        ).set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel("True")
        fig.axes[0].set_ylabel("Predicted")

        fig.savefig(filepath, dpi=250)

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2

        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [
            [want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
            for i in range(want_idx[0].shape[0])
        ]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[
                np.unique(all_matches[:, 1], return_index=True)[1]
            ]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[
                np.unique(all_matches[:, 0], return_index=True)[1]
            ]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if (
                all_matches.shape[0] > 0
                and all_matches[all_matches[:, 0] == i].shape[0] == 1
            ):
                detection_class = detection_classes[
                    int(all_matches[all_matches[:, 0] == i, 1][0])
                ]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or (
                all_matches.shape[0]
                and all_matches[all_matches[:, 1] == i].shape[0] == 0
            ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(" ".join(map(str, self.matrix[i])))


def main(args):
    conf_mat = ConfusionMatrix(CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)
    csv_path = args.submission_csv
    pred_df = pd.read_csv(csv_path)

    file_names = pred_df["image_id"].values.tolist()
    bboxes = pred_df["PredictionString"].values.tolist()
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f"{file_names[i]} empty box")

    gt, new_pred = [], []
    for file_name, bbox in tqdm(zip(file_names, bboxes)):
        new_pred.append([])
        boxes = np.array(str(bbox).split(" "))

        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception("error", "invalid box count")
        for box in boxes:
            new_pred[-1].append(
                [
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                    float(box[5]),
                    float(box[1]),
                    float(box[0]),
                ]
            )

    coco = COCO(GT_JSON_PATH)

    for image_id in coco.getImgIds():
        gt.append([])
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info["id"])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            _gt = [
                float(ann["category_id"]),
                float(ann["bbox"][0]),
                float(ann["bbox"][1]),
                float(ann["bbox"][0]) + float(ann["bbox"][2]),
                (float(ann["bbox"][1]) + float(ann["bbox"][3])),
            ]
            gt[-1].append(_gt)

    for p, g in zip(new_pred, gt):
        conf_mat.process_batch(np.array(p), np.array(g))

    conf_mat.plot(Path(csv_path).parent / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--submission-csv", type=str)
    args = parser.parse_args()
    main(args)
