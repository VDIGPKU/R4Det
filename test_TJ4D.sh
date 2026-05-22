CONFIG_PATH=./configs/r4det/TJ4D-R4Det_det3d_2x4_12e.py
CHECKPOINT_PATH=./work_dirs/TJ4D-R4Det_det3d_2x4_12e-final/epoch_2.pth #./checkpoints/epoch_2.pth ./work_dirs/TJ4D-R4Det_det3d_2x4_12e/epoch_17.pth # # # # # # #
GPUS="8"
PORT=$((RANDOM % 101 + 29600))
#PORT=${PORT:-19501}
#CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/test_vod.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --eval mAP \
    --launcher pytorch ${@:4}
