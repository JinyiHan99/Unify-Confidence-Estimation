import os, torch
import json
import re
import os
from tqdm import tqdm
import random
import argparse
from vllm import LLM, SamplingParams
sys.path.append("/data1/hhx/public/github/Unify-Confidence-Estimation")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")


def dynamic_import(module_name):
    module = __import__(module_name)
    return module


def main(args):

    #load the model
    model = LLM(model=args.model_name, gpu_memory_utilization=0.9)
    print("!!!load the model successfully!")

    #read the data
    datas = read_json(args.dataPath)
    print(f"=========data path:{args.dataPath}")
    infer_data = inference(model, args.size, datas, args.sample_num, args.T)
    infer_data = cal_acc_QA(infer_data)
    infer_data = get_step_1(infer_data)
    cluster_data = get_step1_clusters(infer_data)
    datas = inference_step1(model, args.size, cluster_data, args.sample_num, args.T)
    datas = cal_acc_step1(datas)
    datas = get_Stepn_1(datas)
    datas = inference_stepn_1(model, args.size, datas, args.sample_num, args.T)
    datas = cal_acc_stepn_1(datas)
    cal_acc_A(datas,args.savePath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/dell/lty/ckp/llama_7b_Gsm8k_ToDo_Base/llama_7b_Gsm8k_ToDo_Base_epoch3",
                        help="the name of model")
    parser.add_argument("--dataPath", type=str, default="/data/dell/lty/UCE/test/1.json")
    parser.add_argument("--savePath", type=str, default="/data/dell/lty/UCE/test/2.json")
    parser.add_argument("--sample_num", type=int, default=30, help = "for a question, the number of answer.")
    parser.add_argument("--dataSet", type=str, default="GSM8K", help="[CSQA, GSM8K, TriviaQA]")
    parser.add_argument("--T", type=int, default=1, help="temperature")
    parser.add_argument("--size", type=int, default=4, help="number of data inference in the same batch")

    args = parser.parse_args()
    if (args.dataSet == "CSQA"):
        from utils_CSQA import *
    elif (args.dataSet == "GSM8K"):
        from utils_GSM8K import *
    elif (args.dataSet == "TriviaQA"):
        from utils_TriviaQA import *
    main(args)
