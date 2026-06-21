#!/bin/bash

export CUDA_HOME=/dl_scratch2/ege/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

export VLLM_USE_MULTIPROCESSING_SPAWN=1
export PYTHON_MULTIPROCESSING_METHOD=spawn

temperature=0.0
method=DirectRequest

behaviors_path=./data/behavior_datasets/hex-phi.csv

base_save_dir=./results_hex-phi_temp${temperature}

num_tokens=4096

models=(
    # models before self-evolution
    qwen2.5_3b_coder

    # models after self-evolution
    absolute_zero_3b_coder
)

for model in "${models[@]}"; do
    echo "Processing model: $model"
    python ./scripts/run_pipeline.py \
        --methods $method \
        --models $model \
        --base_save_dir $base_save_dir \
        --behaviors_path $behaviors_path \
        --step all \
        --mode local \
        --max_new_tokens $num_tokens \
        --temperature $temperature \
        --cls_path cais/HarmBench-Llama-2-13b-cls
    
    # Use GPT-4.1 as the api judge
    python -u evaluate_completions_api.py \
        --behaviors_path $behaviors_path \
        --completions_path "./${base_save_dir}/DirectRequest/default/completions/${num_tokens}_tokens/${model}.json" \
        --save_path "./${base_save_dir}/DirectRequest/default/results_gpt4.1/${num_tokens}_tokens/${model}_eval.json" \
        --save_asr_path "./${base_save_dir}/DirectRequest/default/results_gpt4.1/${num_tokens}_tokens/${model}_asr_result.json" \
        --num_tokens $num_tokens
done
