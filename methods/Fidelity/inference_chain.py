import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import requests
import json
import datetime
# import gradio as gr
import torch.nn as nn
from peft import PeftModel
import pdb
import re
import math


from transformers import AutoModelWithLMHead, AutoTokenizer, GenerationConfig
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import NoBadWordsLogitsProcessor

from transformers import AutoModelWithLMHead, T5Tokenizer, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import os

from tqdm import tqdm
import random
import argparse
import pdb
from vllm import LLM, SamplingParams
import ast

import sys
sys.path.append("/Unify-Confidence-Estimation")
from utils import read_json, save_list_to_json, qwen_wrap_overall_instruction_prompt,llama2_wrap_overall_instruction_prompt


device = torch.device("cuda")

def wrap_query(overall_prompt, question, choices):
    ops = ""
    for choice in choices:
        if choice!= "":
            ops += choice + "\n"
    prompt = overall_prompt+"Question: "+ question +"\n"+ops+"Answer: "
    return prompt

def extract_option(text):
    pattern = r'\b[A-Z]\.'
    match = re.search(pattern, text)
    if match:
        # pdb.set_trace()
        op = match.group()[0]
        if op in ['A','B','C','D','E']:
            return op
        return None
    else:
        return None
    
def uncertainty_single_question(ops):
    sum = 0
    num = 10
    ops = ast.literal_eval(ops)
    for item in ops.values():
        sum += item/num * math.log(item/num, 2)
    res = -sum/math.log(5, 2)
    return res


def fidelity_chains(example, t):
    ops = ast.literal_eval(example['ops'])
    uncertaintyQ = uncertainty_single_question(example['ops'])
    print("!!! the uncertainty of the question:", uncertaintyQ)
    op_conf={}
    for op,num in ops.items():
        j = 0
        chainVar = "chain0" # init the original value
        fidelity_op = []
        chain_p = []
        while chainVar in example.keys():
            chain = example[chainVar]
            if op in chain:
                # record its last index
                try:
                    target_index = chain.index(op)+1
                except ValueError:
                    target_index = 5
            fidelity_op.append(target_index)
            chain_p.append(ops[example[chainVar][0]])
            j+=1
            chainVar = f"chain{j}"
            # pdb.set_trace()
        print("!!!fidelity_op:",fidelity_op)
        print("!!!sample p:",chain_p)
    
        fidelity_op_sum = 0
        for k in range(len(fidelity_op)):
            fidelity_op_sum += pow(t,fidelity_op[k])
        res = 0
        for k in range(len(fidelity_op)):
            res += chain_p[k] / 10 * pow(t,fidelity_op[k])/fidelity_op_sum 
            # pdb.set_trace()
        print("!!!op的fidelity:", res)
        conf = (1-uncertaintyQ)*res
        op_conf[op] = conf
    # pdb.set_trace()
    print("!!final answer:", op_conf)


    return op_conf

@torch.no_grad()
def get_response(model, prompts):
    generation_config = SamplingParams(
                # temperature=0.5,
                top_p=0.8,
                top_k=50,
                # frequency_penalty = 1.1,
                max_tokens= 2048,
                temperature = 1,
                stop="\n"
        )
    with torch.no_grad():
        outputs = model.generate(prompts,generation_config)
    responses =[]
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        # pdb.set_trace()
        response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '').replace("<pad> ","")
        responses.append(response)
    return prompt, responses

def main(args):
    #1. load the model 
    # 使用vllm 来加载模型
    model = LLM(model=args.model_path, gpu_memory_utilization=0.7)
    print("!!!load the model successfully!")
    #2. read the data
    # data = read_json(args.data_path)
    data = read_json(args.data_path)
    print(f"=========data path:{args.data_path}")

# here is the 5-shot prompt
    train = read_json("/data/commonsenseQA/train.json")
    ans_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    global_prompt = ""
    for example in train[:5]:
        choices = example['lables']
        ops = ""
        for choice in choices:
            ops += choice + "\n"
        prompt = "Question: " + example['question']+"\n" + ops + "Answer: "+ choices[ans_map[example['std_ans']]]+"\n"
        global_prompt += prompt
    


    save_data = []
    save_path = args.save_path
    
    for example in tqdm(data):
        prompts = []
        ops_dic = ast.literal_eval(example['ops'])
        ops_list = list(ops_dic.keys())
        chains_num = len(ops_list)
        labels =  example['lables']
        for i in range(chains_num):      
            labels_temp = list(labels)
            replace_op = ops_list[i]
            chains = []
            op_generation = ""
            chains.append(replace_op)
            replace_op_index = ans_map[replace_op]
            labels_temp[replace_op_index] = f"{replace_op}. All other options are wrong."
            while op_generation!= replace_op:
                #if the chosen option is not equal the target index, this option will be removed
                print("!!!op_generation:",op_generation)
                if op_generation is not None and op_generation.strip():
                    remove_op_index = ans_map[op_generation]
                    # del(labels_temp[remove_op_index])
                    labels_temp[remove_op_index] = ""
                    # pdb.set_trace() 
                prompt = wrap_query(global_prompt, example['question'], labels_temp)
                    # print("!!!!!!",prompt)
                prompts.append(prompt)
                prompt, responses = get_response(model, prompts)
                op_generation = extract_option(responses[0])
                # if op_generation
                flag = 0
                for op_temp in labels_temp:
                    if op_temp!="" and op_generation == op_temp[0]:
                        flag =1
                if flag ==0:
                    break
                chains.append(op_generation)
            # if chains[-1]==chains[0]:
            #     del(chains[-1])
            print("!!!chains:",chains)
            var = f"chain{i}"
            example[var] = chains
            save_data.append(example)
            save_list_to_json(save_data, save_path)

    res = []
    for example in save_data:
        op_conf = fidelity_chains(example,2)
        example['conf'] = json.dumps(op_conf)
        res.append(example)

    save_list_to_json(res, save_path)
        


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models--meta-llama--Llama-2-7b-chat-hf/",
                        help="the name of model")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()
    main(args)


