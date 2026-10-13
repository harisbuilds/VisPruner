#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/path/to/checkpoint}"  # [LOG] can be set from the environment
DATA_DIR="${DATA_DIR:-/path/to/dataset}"  # [LOG] can be set from the environment

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"

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

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/gqa/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/${RUN}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --important_ratio ${RATIO} \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        "${EXTRA_ARGS[@]}" &
done

wait

GQA_DIR="./playground/data/eval/gqa/data"
output_file=./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/${RUN}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/${RUN}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# [LOG] The official scorer needs every question, so skip it for --limit subsets
if [ -n "${LIMIT}" ]; then
    echo "[LOG] --limit ${LIMIT}: skipping official scoring. Answers in ${output_file}"
    exit 0
fi

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${GQA_DIR}/testdev_balanced_predictions.json

cd ${GQA_DIR}
python eval/eval.py \
    --path ${DATA_DIR}/gqa/data/questions \
    --tier testdev_balanced
