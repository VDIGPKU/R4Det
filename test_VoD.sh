CONFIG_PATH=./configs/r4det/vod-R4Det_det3d_2x4_12e.py
CHECKPOINT_PATH=./work_dirs/vod-R4Det_det3d_2x4_12e/iter_2.pth #epoch_9.pth  #./checkpoints/epoch_3.pth#./xzy/epoch_3.pth #. # # #/output/vod/epoch_3.pth # #./checkpoints/FINAL-VoD.pth # #
OUTPUT_NAME=pts_bbox #vod-R4Det
PRED_RESULTS=./work_dirs/submissions/$OUTPUT_NAME  #./tools_det3d/view-of-delft-dataset/pred_results/$OUTPUT_NAME

GPUS="8"
PORT=$((RANDOM % 101 + 29600))
#PORT=${PORT:-19521}
#CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NATTEN_LOG_LEVEL="critical" python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/test_vod.py \
    --format-only \
    --eval-options submission_prefix=$PRED_RESULTS \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --launcher pytorch ${@:4}

python tools/view-of-delft-dataset/FINAL_EVAL.py \
--pred_results $PRED_RESULTS
