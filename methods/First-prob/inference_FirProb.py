import os
import time
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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
    score = ""
    response = ""
    num = 0
    scores_list = []
    for tensor in scores:
        top_5_value, _ = torch.topk(tensor, k=5)
        softmax_tensor = F.softmax(top_5_value, dim=1)
        scores_list.append(softmax_tensor.tolist())
        if (num == 0):
            score = softmax_tensor.tolist()
        num += 1

    for i, output_sequence in enumerate(generation_output.sequences):
        output_text = tokenizer.decode(output_sequence, skip_special_tokens=False)  # 不然数量对不上
        start_index = output_text.find("[/INST]")
        response = output_text[start_index+7:]
        response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '').replace("<pad> ", "")

    return response, score[0][0], scores_list

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
    for example in tqdm(data):
        instr = wrapQuery(example)
        query = instr
        prompt = query
        response, score, scores_list = get_response(model, tokenizer, prompt)
        example['response'] = response
        example['score'] = score
        save_data.append(example)
        save_list_to_json(save_data, args.savePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/dell/lty/ckp/llama_7b_ComQA_Splite_Conf/llama_7b_ComQA_Splite_Conf_epoch2",
                        help="the name of model")
    parser.add_argument("--lora_weights", type=str, default="",
                        help="if you use lora model, please fill this value, or it is empty")
    parser.add_argument("--dataPath", type=str,
                        default="/data/dell/lty/UCE/test/test.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/test_FirToken.json")
    args = parser.parse_args()
    main(args)