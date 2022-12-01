# command

## wandb

```shell
wandb artifact cache cleanup 100
```

## linux

```shell
df -h
du -ah -d 1 | sort -h
```

## mmdetection

```shell
python mmdetection/tools/train.py mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug.py
python mmdetection/tools/test.py mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug.py work_dirs/cascade_rcnn_swinL_aug/latest.pth
```

## formatting

```shell
black . && isort .
```

## conda

```shell
conda create --clone detection --name detection2
conda env remove --name env_name

conda list --explicit > spec-file.txt
conda create --name new_env --file spec-file.txt

conda env export > environment.yml
conda env create -f environment.yml

pip freeze > environment-pip-freeze.txt
conda list --revisions > environment-revisions.txt

conda clean --yes --all
```

## pre-commit

```shell
pre-commit install
pre-commit uninstall
```
