import os, torch
import json
import re
import os
import sys
from tqdm import tqdm
import random
import argparse
from vllm import LLM, SamplingParams
sys.path.append("/data1/hhx/public/github/Unify-Confidence-Estimation")
from utils_CSQA import *
from config import prompt_templates
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

prompt_description = prompt_templates['CSQA_Multistep']


device = torch.device("cuda")


if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))

    print(f"Total GPUs available: {len(device_ids)}")

    for i in device_ids:
        device = torch.device(f"cuda:{i}")
        print(f"GPU Device ID {i}: {torch.cuda.get_device_name(device)}")
else:
    print("No GPU available.")

generation_config_hhx = SamplingParams(
        # temperature=0.5,
        top_p=0.8,
        top_k=50,
        frequency_penalty = 1.1,
        max_tokens= 2048,
        # logits_processors = [control_T],??
        # use_beam_search = True,
        # num_beams=1,
        # do_sample = True,
        # repetition_penalty=1.1,
        # length_penalty = 1.0,
        # max_length = 2048,
        # max_new_tokens = 1024,
        # early_stopping = True,
)



def main(args):
    # 1. load the llm
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
            prompt = wrapQuery(example, prompt_description)
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
                        default="/data/dell/lty/ckp/llama_7b_ComQA_Cot_Base/llama_7b_ComQA_Cot_Base_epoch3",
                        help="the name of model")
    parser.add_argument("--dataPath", type=str, default="/data/dell/lty/UCE/test/test.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/test_Multi.json", help=" save as json ")
    parser.add_argument("--sample_num", type=int, default=1, help="for a question, the number of answer.")
    parser.add_argument("--T", type=int, default=1, help="temperature")
    parser.add_argument("--size", type=int, default=4, help="number of data inference in the same batch")

    args = parser.parse_args()
    main(args)
