from vllm import LLM, SamplingParams
import os
import re
import sys
import torch
from tqdm import tqdm
import json
import random
import pickle
import argparse

sys.path.append("/data1/hhx/public/github/Unify-Confidence-Estimation")
from utils_CSQA import *
from config import prompt_templates
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

prompt_description = prompt_templates['CSQA_Verb']

device = torch.device("cuda")


if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))

    print(f"Total GPUs available: {len(device_ids)}")

    for i in device_ids:
        device = torch.device(f"cuda:{i}")
        print(f"GPU Device ID {i}: {torch.cuda.get_device_name(device)}")
else:
    print("No GPU available.")



@torch.no_grad()
def get_response(model, prompts, T, output_num):
    generation_config = SamplingParams(
        # temperature=0.5,
        top_p=0.8,
        top_k=50,
        # frequency_penalty = 1.1,
        max_tokens=2048,
        temperature=T,
        n=output_num
    )
    outputs = model.generate(prompts, generation_config)

    responses = []
    for output in outputs:
        prompt = output.prompt
        single_response = output.outputs
        single_response_list = []
        for i in range(len(single_response)):
            response = single_response[i].text
            response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '').replace("<pad> ", "")
            single_response_list.append(response)
        responses.append(single_response_list)
    return prompt, responses


def main(args):
    #1. load the llm
    model = LLM(model=args.model_name, gpu_memory_utilization=0.9)
    print("!!!load the model successfully!")
    # 2. read the data
    data = read_json(args.dataPath)
    # data = read_json_line(args.dataPath)
    print(f"=========data path:{args.dataPath}")

    save_data = []
    savePath = args.savePath
    size = args.size
    for i in tqdm(range(0, len(data), size)):
        group = data[i: i + size]  # current examples
        prompts = []
        for example in group:
            prompt = wrapQuery(example,prompt_description)
            prompts.append(prompt)
        prompt, responses = get_response(model, prompts, args.T, args.sample_num)
        for j in range(len(group)):
            example = group[j]
            response = responses[j]
            example['response'] = response
            save_data.append(example)
        # pdb.set_trace()
        save_list_to_json(save_data, savePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="/data/cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/717c4c9e612e583993155c2b586332e301b61f55",
                        help="the name of model")
    parser.add_argument("--dataPath", type=str, default="/data/dell/lty/UCE/test/test.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/test_Verb2.json", help=" save as json ")
    parser.add_argument("--sample_num", type=int, default=1, help="for a question, the number of answer.")
    parser.add_argument("--T", type=int, default=1, help="temperature")
    parser.add_argument("--size", type=int, default=4, help="number of data inference in the same batch")

    args = parser.parse_args()
    main(args)


