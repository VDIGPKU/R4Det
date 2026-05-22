#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=$((RANDOM + 10000))
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUMBA_CUDA_FORCE_PTX_VERSION=82 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NATTEN_LOG_LEVEL="critical" python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train_vod.py \
    --config $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
