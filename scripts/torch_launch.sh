#!/usr/bin/env bash
# Script to start PyTorch training in one of two modes to choose from:
# DataParallel and DistributedDataParallel, - based on the configuration file

# Example usege:
# ./train.sh \
#   --file-path=/project/scripts/train.py \
#   --cuda-visible-devices=0,1,2 \
#   --is-distributed=true \
#   --positional1=any \
#   --positional2=any


IFS=","
POSITIONAL=()

# Args parsing
for i in "$@"; do
    case "$i" in
        -fp=*|--file-path=*)
            FILE_PATH="${i#*=}"
            shift
        ;;
        -cvd=*|--cuda-visible-devices=*)
            CUDA_VISIBLE_DEVICES="${i#*=}"
            shift
        ;;
        -id=*|--is-distributed=*)
            IS_DISTRIBUTED="${i#*=}"
            shift
        ;;
        *)
            POSITIONAL+=("$i")
            shift
    esac
done


if [ $IS_DISTRIBUTED = true ]; then
    read -ra GPUS <<<"$CUDA_VISIBLE_DEVICES"
    GPU_COUNT="${#GPUS[@]}"

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python -m torch.distributed.launch \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$GPU_COUNT \
        $FILE_PATH ${POSITIONAL[@]}

elif [ $IS_DISTRIBUTED = false ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python $FILE_PATH ${POSITIONAL[@]}

else
    echo "Invalid option: --is-distributed=$IS_DISTRIBUTED"
    exit 1
fi
