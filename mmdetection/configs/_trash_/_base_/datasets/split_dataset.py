import json
import random

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SEED = 2022
TOP_DIR = "/opt/ml/dataset/"
TRAIN_JSON_FILE = TOP_DIR + "train_all.json"
# 기존 /opt/ml/dataset/train.json 파일을 복제해서 파일명을 train_all.json으로 바꾸세요


def random_split(input_json, val_ratio=0.2, random_seed=SEED):
    random.seed(random_seed)

    with open(input_json, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    image_ids = [x.get("id") for x in images]
    image_ids.sort()
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * val_ratio)
    num_train = len(image_ids) - num_val

    image_ids_val, image_ids_train = set(image_ids[:num_val]), set(image_ids[num_val:])

    train_images = [x for x in images if x.get("id") in image_ids_train]
    val_images = [x for x in images if x.get("id") in image_ids_val]

    train_annos = [x for x in annotations if x.get("image_id") in image_ids_train]
    val_annos = [x for x in annotations if x.get("image_id") in image_ids_val]

    train_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": train_images,
        "categories": data["categories"],
        "annotations": train_annos,
    }

    val_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "images": val_images,
        "categories": data["categories"],
        "annotations": val_annos,
    }

    output_train_json = TOP_DIR + f"train_rand_{random_seed}_{1-val_ratio}.json"
    output_val_json = TOP_DIR + f"val_rand_{random_seed}_{val_ratio}.json"

    print(f"write {output_train_json}")
    with open(output_train_json, "w") as train_writer:
        json.dump(train_data, train_writer)

    print(f"write {output_val_json}")
    with open(output_val_json, "w") as val_writer:
        json.dump(val_data, val_writer)


def stratified_group_kfold(dataset_dir: str = TRAIN_JSON_FILE, save_dir: str = TOP_DIR):
    """
    StratifiedGroupFKold 방식으로 dataset을 나누는 함수

    Args:
        dataset_dir : dataset(train_all.json)의 경로
        save_dir : 저장 경로

    Note:
        이 함수에서 TOP_DIR이 저장경로로 사용됨
        함수 실행 시 저장 경로에 train,val의 json 파일이 생성됨

    """

    with open(f"{TRAIN_JSON_FILE}") as f:
        data = json.load(f)

    var = [(ann["image_id"], ann["category_id"]) for ann in data["annotations"]]
    X = np.ones((len(data["annotations"]), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

    idx = 0

    for train_idx, val_idx in cv.split(X, y, groups):
        print("TRAIN:", groups[train_idx])
        print("      ", y[train_idx])
        print(" TEST:", groups[val_idx])
        print("      ", y[val_idx])

        # train, val dataset의 겹치는 부분 확인
        print(set(groups[train_idx]) & set(groups[val_idx]))

        train = dict()
        val = dict()

        # make train dataset
        train["images"] = [data["images"][i] for i in set(groups[train_idx])]
        train["categories"] = data["categories"]
        train["annotations"] = [data["annotations"][i] for i in train_idx]
        # 저장 경로
        with open(f"{TOP_DIR}train_groupk_{idx}.json", "w") as make_file:
            json.dump(train, make_file, indent="\t")

        # make validation dataset
        val["images"] = [data["images"][i] for i in set(groups[val_idx])]
        val["categories"] = data["categories"]
        val["annotations"] = [data["annotations"][i] for i in val_idx]
        # 저장 경로
        with open(f"{TOP_DIR}val_groupk_{idx}.json", "w") as make_file2:
            json.dump(val, make_file2, indent="\t")

        idx += 1


if __name__ == "__main__":
    random_split(TRAIN_JSON_FILE)
    random_split(TRAIN_JSON_FILE, val_ratio=0.3)

    # stratified_group_kfold(TRAIN_JSON_FILE, TOP_DIR)
