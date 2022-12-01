# WARNING: Not finished yet

import math
from pathlib import Path
from typing import Union

import bbox_visualizer as bbv
import cv2
import numpy as np
import plotly.express as px
import typer
from mmdet.core.visualization import get_palette, imshow_det_bboxes
from PIL import Image

DATASET_PATH = Path("/opt/ml/dataset")
TRAIN_JSON = DATASET_PATH / "train.json"

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

COLOR_PALETTE = [
    (3, 3, 204),
    (204, 0, 204),
    (255, 254, 2),
    (51, 204, 0),
    (1, 128, 1),
    (152, 0, 102),
    (4, 188, 243),
    (94, 179, 0),
    (233, 18, 36),
    (108, 23, 170),
]


def convert_pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def convert_cv2_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def add_bbox_and_text(img, text, bbox, color) -> None:
    # https://stackoverflow.com/questions/52846474/how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python

    FONT_SCALE = 5e-4  # Adjust for larger font size in all images
    THICKNESS_SCALE = 9e-4  # Adjust for larger thickness in all images
    TEXT_Y_OFFSET_SCALE = 5e-3  # Adjust for larger Y-offset of text and bounding box

    height, width, _ = img.shape
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        img,
        text,
        (x_min, y_min - int(height * TEXT_Y_OFFSET_SCALE)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=min(width, height) * FONT_SCALE,
        thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
        color=color,
    )


def draw_label_bbox_on_image(img: Image.Image, predict_list: np.ndarray) -> Image.Image:
    cv_img = convert_pil_to_cv2(img)
    for _class, score, *bbox in predict_list:
        label = CLASSES[int(_class)]
        color = COLOR_PALETTE[int(_class)]
        bbox = [int(float(x)) for x in bbox]
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(cv_img, (x_min, y_min), (x_max, y_max), color, 2)
        add_bbox_and_text(cv_img, f"{label} {float(score):.3f}", bbox, color)

    return convert_cv2_to_pil(cv_img)


def open_csv_files(csv_list_file: Union[str, Path]):
    """csv 파일 목록.txt 을 열어서 submission.csv 파일들을 열어서 그 내용을 return하는 함수

    Args:
        csv_list_file (Union[str, Path]): 1개 이상의 submission.csv이 줄바꿈으로 이루어진 파일 경로
    """
    if isinstance(csv_list_file, str):
        csv_list_file = Path(csv_list_file)

    assert csv_list_file.exists()

    with open(csv_list_file, "r") as f:
        csv_file_list = f.readlines()

    assert len(csv_file_list) > 0

    for csv_file in csv_file_list:
        pass


def main(csv_list_file: Union[str, Path] = "visualize_csv.txt") -> None:
    """csv 파일 목록 읽어서 이미지 별 bbox 그려서 비교하기

    모든 이미지에 대해서 csv 파일별 bbox annotated image를 subplot으로 이어 붙인 이미지를 저장한다.

    csv 파일이 3개라면, test 이미지 중 test/0001.jpg 하나를 예시로 들자면,
    csv 파일별 bbox를 그린 이미지(총 3장)을 바둑판식으로 이어 붙여서 show_dir/0001.jpg로 내보낸다.

    Args:
        csv_list_file (Union[str, Path], optional): _description_. Defaults to "visualize_csv.txt".
    """

    with open(csv_list_file, "r") as f:
        csv_lines = f.readlines()

    for csv_line in csv_lines[1:]:
        image_file, predict_str = csv_line.split(",")
        predict_list = np.reshape(predict_str.split(), (-1, 6))
        img = Image.open(DATASET_PATH / image_file)
        annotated_img = draw_label_bbox_on_image(img, predict_list)


if __name__ == "__main__":
    typer.run(main)
