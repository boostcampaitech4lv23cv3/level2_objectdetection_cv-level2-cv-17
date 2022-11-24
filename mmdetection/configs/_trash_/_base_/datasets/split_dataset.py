import json
import random

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


if __name__ == "__main__":
    random_split(TRAIN_JSON_FILE)
    random_split(TRAIN_JSON_FILE, val_ratio=0.3)
