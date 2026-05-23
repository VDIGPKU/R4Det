CONFIG_PATH=./configs/
CHECKPOINT_PATH=./work_dirs/ 
OUTPUT_NAME=pts_bbox
PRED_RESULTS=./work_dirs/submissions/$OUTPUT_NAME 

GPUS="8"
PORT=$((RANDOM % 101 + 29600))
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
