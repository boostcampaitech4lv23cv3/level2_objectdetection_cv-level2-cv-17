for i in 0 1 2 3 4
do
    echo ""
    echo "============================================================================================================="
    echo "wandb artifact cache cleanup 100"
    wandb artifact cache cleanup 100

    echo ""
    echo "============================================================================================================="
    echo "python mmdetection/tools/train.py mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug$i.py"
    python mmdetection/tools/train.py "mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug$i.py"

    echo ""
    echo "============================================================================================================="
    echo "wandb artifact cache cleanup 100"
    wandb artifact cache cleanup 100

    echo ""
    echo "============================================================================================================="
    echo "python mmdetection/tools/test.py mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug$i.py" "work_dirs/cascade_rcnn_swinL_aug$i/latest.pth"
    python mmdetection/tools/test.py "mmdetection/configs/_trash_/_base_/models/swin/cascade_rcnn_swinL_aug$i.py" "work_dirs/cascade_rcnn_swinL_aug$i/latest.pth"
done