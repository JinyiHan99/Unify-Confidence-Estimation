import os, torch
import json
import re
import os
import sys
from tqdm import tqdm
import random
import argparse
import pickle
from vllm import LLM, SamplingParams

sys.path.append("/data1/hhx/public/github/Unify-Confidence-Estimation")
from utils_CSQA import *
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

device = torch.device("cuda")


@torch.no_grad()
def get_response(model, prompts, T, output_num):
    generation_config = SamplingParams(
                # temperature=0.5,
                top_p=0.8,
                top_k=50,
                # frequency_penalty = 1.1,
                max_tokens= 2048,
                temperature = T,
                n = output_num
        )
    outputs = model.generate(prompts, generation_config)

    responses =[]
    for output in outputs:
        prompt = output.prompt
        single_response = output.outputs
        single_response_list = []
        for i in range(len(single_response)):
            response = single_response[i].text
            response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '').replace("<pad> ","")
            single_response_list.append(response)
        responses.append(single_response_list)
    return prompt, responses

def main(args):
    #1. load the model
    model = LLM(model=args.model_name, gpu_memory_utilization=0.9)

    print("!!!load the model successfully!")


    #2. read the data
    data = read_json(args.dataPath)
    # data = read_json_line(args.dataPath)
    print(f"=========data path:{args.dataPath}")

    newData = []
    savePath = args.savePath
    size = args.size
    for i in tqdm(range(0, len(data), size)):
        group = data[i: i + size]  # current examples
        prompts = []
        for example in group:
            prompt = wrapQuery(example)
            prompts.append(prompt)
        prompt, responses = get_response(model, prompts, args.T, args.sample_num)
        for j in range(len(group)):
            example = group[j]
            response = responses[j]
            example['response'] = response
            newData.append(example)

    newData = cal_acc_TorF(newData)

    dataSave = []
    for data in newData:
        for flag in data["TorF"]:
            row1 = {}
            instr, output = wrapQuery_pro_out(data, flag)
            row1["instruction"] = instr
            row1["input"] = ""
            row1["output"] = output
            row1 = [row1]
            dataSave.append(row1)
    save_list_to_json(dataSave,"/data/dell/lty/UCE/test/test_PIK.json")
    with open(savePath, "wb") as f:
        pickle.dump(dataSave, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/dell/lty/ckp/llama_7b_ComQA_Cot_Base/llama_7b_ComQA_Cot_Base_epoch3",
                        help="the name of model")
    parser.add_argument("--dataPath", type=str, default="/data/dell/lty/UCE/test/test.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/test_PIK.pkl", help =" save as pkl ")
    parser.add_argument("--sample_num", type=int, default=30, help = "for a question, the number of answer.")
    parser.add_argument("--T", type=int, default=1, help="temperature")
    parser.add_argument("--size", type=int, default=4, help="number of data inference in the same batch")

    args = parser.parse_args()
    main(args)
