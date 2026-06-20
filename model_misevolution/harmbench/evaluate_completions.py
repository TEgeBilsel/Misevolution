import multiprocessing
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import json
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--save_asr_path", type=str, default=None,
                        help="The path for saving asr(attack successful rate)")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side="right"

    # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
    for behavior_id, completion_list in completions.items():
        for completion in completion_list:
            generation = completion['generation'].split('</think>')[-1]

            tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
            clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            completion['generation'] = clipped_generation

    # ========== evaluate completions ========== #
    class HFClassifierWrapper:
        def __init__(self, model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.truncation_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

        def generate(self, prompts, sampling_params=None, **kwargs):
            if isinstance(prompts, str):
                prompts = [prompts]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[:, inputs["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

            class Obj:
                pass

            wrapped = []
            for text in texts:
                o = Obj()
                oo = Obj()
                oo.text = text
                o.outputs = [oo]
                wrapped.append(o)
            return wrapped

    cls = HFClassifierWrapper(args.cls_path)
    cls_params = None

    from eval_utils import compute_results_hashing, compute_results_classifier, compute_results_advbench


    results = {}
    for behavior_id, data in tqdm(completions.items()):
        if behavior_id not in behaviors:
            print(f"Behavior {behavior_id} not found in behaviors file; skipping")
            continue
        behavior_dict = behaviors[behavior_id]
        tags = behavior_dict['Tags'].split(', ')

        if 'hash_check' in tags:
            current_results = compute_results_hashing(behavior_dict, data)
        else:
            current_results = compute_results_classifier(behavior_dict, data, cls, cls_params)
        
        if args.include_advbench_metric:
            current_results = compute_results_advbench(current_results, data)

        results[behavior_id] = current_results

    success_rates = []
    asr_results = {
        "per_behavior": {},
        "average_asr": None
    }

    for i, (behavior_id, data) in enumerate(results.items()):
        success_mean = np.mean([d['label'] == 1 for d in data])
        print(f"{i}. {behavior_id} === average asr: {success_mean}")
        success_rates.append(success_mean)
        asr_results["per_behavior"][behavior_id] = success_mean
    
    avg_asr = np.mean(success_rates)
    asr_results["average_asr"] = avg_asr
    print("Average ASR: ", avg_asr)

    # 保存详细结果
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(results, file, indent=4)
    
    # 保存ASR结果
    os.makedirs(os.path.dirname(args.save_asr_path), exist_ok=True) if os.path.dirname(args.save_asr_path) else None
    with open(args.save_asr_path, 'w') as file:
        json.dump(asr_results, file, indent=4)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()

# import multiprocessing
# import transformers
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# import json
# import argparse
# import os
# import csv
# from tqdm import tqdm 
# import numpy as np


# def parse_args():
#     parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
#     parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
#                         help="The name or path of the classifier for computing ASR")
#     parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
#                         help="The path to the behaviors file")
#     parser.add_argument("--completions_path", type=str,
#                         help="The path to the completions to evaluate")
#     parser.add_argument("--save_path", type=str,
#                         help="The path for saving results")
#     parser.add_argument("--save_asr_path",type=str,
#                         help="The path for saving asr(attack successful rate)")
#     parser.add_argument("--include_advbench_metric", action="store_true",
#                         help="Whether to include the AdvBench refusal metric")
#     parser.add_argument("--num_tokens", type=int, default=512,
#                         help="The number of tokens to evaluate")
#     args = parser.parse_args()
#     return args


# def main():
#     # ========== load arguments and config ========== #
#     args = parse_args()
#     print(args)

#     # ========== load behaviors (for tags and context strings) ========== #
#     with open(args.behaviors_path, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         behaviors = list(reader)
    
#     # convert to dictionary mapping behavior ID field to behavior dict
#     behaviors = {b['BehaviorID']: b for b in behaviors}

#     # ========== initialize results ========== #
#     with open(args.completions_path, 'r') as f:
#         completions = json.load(f)

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
#     tokenizer.truncation_side="right"

#     # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
#     for behavior_id, completion_list in completions.items():
#         for completion in completion_list:
#             generation = completion['generation'].split('</think>')[-1]

#             tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
#             clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
#             completion['generation'] = clipped_generation

#     # ========== evaluate completions ========== #

#     cls = LLM(model=args.cls_path, tensor_parallel_size=8)

#     cls.llm_engine.tokenizer.tokenizer.truncation_side="left"

#     cls_params = SamplingParams(temperature=0.0, max_tokens=1)

#     from eval_utils import compute_results_hashing, compute_results_classifier, compute_results_advbench


#     results = {}
#     for behavior_id, data in tqdm(completions.items()):
#         if behavior_id not in behaviors:
#             print(f"Behavior {behavior_id} not found in behaviors file; skipping")
#             continue
#         behavior_dict = behaviors[behavior_id]
#         tags = behavior_dict['Tags'].split(', ')

#         if 'hash_check' in tags:
#             current_results = compute_results_hashing(behavior_dict, data)
#         else:
#             current_results = compute_results_classifier(behavior_dict, data, cls, cls_params)
        
#         if args.include_advbench_metric:
#             current_results = compute_results_advbench(current_results, data)

#         results[behavior_id] = current_results

#     success_rates = []

#     for i, (behavior_id, data) in enumerate(results.items()):
#         success_mean = np.mean([d['label'] == 1 for d in data])
#         print(f"{i}. {behavior_id} === average asr: {success_mean}")
#         success_rates.append(success_mean)
    
#     print("Average ASR: ", np.mean(success_rates))

#     os.makedirs(os.path.dirname(args.save_asr_path), exist_ok=True) if os.path.dirname(args.save_asr_path) else None
#     with open(args.save_asr_path, 'w') as file:
#         json.dump("Average ASR: ", np.mean(success_rates))

#     # Make dirs to output_file if not yet exist
#     os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
#     with open(args.save_path, 'w') as file:
#         json.dump(results, file, indent=4)

# if __name__ == "__main__":
#     multiprocessing.set_start_method('spawn', force=True)
#     main()
