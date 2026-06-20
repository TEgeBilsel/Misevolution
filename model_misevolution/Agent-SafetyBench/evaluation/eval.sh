


models=(
  qwen2.5-3b-inst
)


for model in "${models[@]}"; do
  echo "===== Running with model=$model ====="
  CUDA_VISIBLE_DEVICES=0 python -u eval.py \
    --model_name $model \
    --greedy 1 \
    --regen_exceed 1 \
    --extra_info ""
done