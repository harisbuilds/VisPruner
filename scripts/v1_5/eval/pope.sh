#!/bin/bash

CKPT_DIR="${CKPT_DIR:-/path/to/checkpoint}"  # [LOG] can be set from the environment
DATA_DIR="${DATA_DIR:-/path/to/dataset}"  # [LOG] can be set from the environment

CKPT="llava-v1.5-7b"
SPLIT="llava_pope_test"

TOKEN=${1}
RATIO=${2}

# [LOG] Optional extra args (--log_dir DIR, --blank_image, --limit N) are passed to model_vqa_loader.
# They also go into the run name so these runs don't overwrite the default answers.
EXTRA_ARGS=("${@:3}")
RUN="r_${RATIO}"
LIMIT=""
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        --blank_image) RUN="${RUN}_blank" ;;
        --limit) LIMIT="${EXTRA_ARGS[$((i+1))]}"; RUN="${RUN}_first${LIMIT}" ;;
        --log_dir) RUN="${RUN}_log" ;;
    esac
done

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/${RUN}.jsonl \
    --visual_token_num ${TOKEN} \
    --important_ratio ${RATIO} \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    "${EXTRA_ARGS[@]}"

# [LOG] eval_pope.py matches answers to labels by line order, so it is wrong on --limit subsets
if [ -n "${LIMIT}" ]; then
    echo "[LOG] --limit ${LIMIT}: skipping official scoring"
    exit 0
fi

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/${RUN}.jsonl
