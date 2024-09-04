import os
import sys
import time
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from tqdm import tqdm
import json
import random
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("/data1/hhx/public/github/Unify-Confidence-Estimation")
from utils_CSQA import *
device = torch.device("cuda")


if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))

    print(f"Total GPUs available: {len(device_ids)}")

    for i in device_ids:
        device = torch.device(f"cuda:{i}")
        print(f"GPU Device ID {i}: {torch.cuda.get_device_name(device)}")
else:
    print("No GPU available.")

generation_config = GenerationConfig(
    temperature=1,
    top_p=1,  # do_sample = True
    top_k=50,  # do_sample = True
    do_sample=True,
    frequency_penalty=1.1,
    max_tokens=2048,
    max_length=2048,
    max_new_tokens=1024,
)

def calculate_entropy(probabilities):
    if not probabilities:
        return 0
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def get_response(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.convert_tokens_to_ids('<s>'),
            pad_token_id=tokenizer.eos_token_id,
            min_length=input_ids.shape[1] + 1
        )
    scores = generation_output['scores']
    score_step1 = 0
    score_stepn_1 = 0
    score_all = 0
    response = ""
    num = 0
    scores_list = []
    sum = len(scores)
    for tensor in scores:
        top_5_value, _ = torch.topk(tensor, k=5)
        softmax_tensor = F.softmax(top_5_value, dim=1)
        score = softmax_tensor.tolist()
        if num < sum/3:
            score_step1 += score[0][0]
        if num < sum*2/3:
            score_stepn_1 += score[0][0]
        score_all += score[0][0]
        scores_list.append(score[0][0])
        num += 1
    score_step1 = score_step1/(sum/3)
    score_stepn_1 = score_stepn_1/(sum*2/3)
    score_all = score_all/sum
    entropy = calculate_entropy(scores_list)
    for i, output_sequence in enumerate(generation_output.sequences):
        output_text = tokenizer.decode(output_sequence, skip_special_tokens=False)  # 不然数量对不上
        start_index = output_text.find("[/INST]")
        response = output_text[start_index+7:]
        response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '').replace("<pad> ", "")

    return response, score_step1, score_stepn_1, score_all, entropy

def main(args):
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    if args.lora_weights:
        LORA_WEIGHTS = args.lora_weights
        model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
        print(f"=======model-lora loaded:{LORA_WEIGHTS}")
    else:
        print(f"=======model-full loaded:{model_name}")
    data = read_json(args.dataPath)
    print(f"=========data path:{args.dataPath}")
    save_data = []
    max_entropy = 0
    min_entropy = 1000000000
    for example in tqdm(data):
        instr = wrapQuery(example)
        query = instr
        prompt = query
        response, score_step1, score_stepn_1, score_all, entropy = get_response(model, tokenizer, prompt)
        example['response'] = response
        example['score_step1'] = score_step1
        example['score_stepn_1'] = score_stepn_1
        example['score_all'] = score_all
        example['entropy'] = entropy
        if(max_entropy < entropy):
            max_entropy = entropy
        if (min_entropy > entropy):
            min_entropy = entropy
        save_data.append(example)
        save_list_to_json(save_data, args.savePath)

    for example in save_data:
        example["conf"] = ((example['score_step1'] + example['score_stepn_1'])/2) *(1-(example['entropy'] - min_entropy) / (max_entropy - min_entropy))

    save_list_to_json(save_data, args.savePath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/dell/lty/ckp/llama_7b_ComQA_Splite_Conf/llama_7b_ComQA_Splite_Conf_epoch2",
                        help="the name of model")
    parser.add_argument("--lora_weights", type=str, default="",
                        help="if you use lora model, please fill this value, or it is empty")
    parser.add_argument("--dataPath", type=str,
                        default="/data/dell/lty/UCE/test/test.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/test_LECO2.json")
    args = parser.parse_args()
    main(args)