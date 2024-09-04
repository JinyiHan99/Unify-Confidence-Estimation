import os, torch

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import json
import pdb
from tqdm import tqdm

device = torch.device("cuda")

import sys

sys.path.append("/data1/hhx/public/github")
from utils import read_json, save_list_to_json, qwen_wrap_overall_instruction_prompt, \
    llama2_wrap_overall_instruction_prompt
from config import prompt_templates


def extract_step_answer(text):
    text = text.replace(" Answer:\n", "")
    lines = text.split('\n')
    step_n_1 = ""
    step_1 = ""
    for i, line in enumerate(lines, start=1):
        if i == 1:
            step_1 += f"{line}\n"
        if i <= len(lines) - 2:
            step_n_1 += f"{line}\n"
    return step_1, step_n_1


@torch.no_grad()
def get_response(tokenizer, model, lora_name, prompt, model_name):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
    if model_name == "qwen":
        text = qwen_wrap_overall_instruction_prompt(tokenizer, prompt)
    if model_name == "llama2":
        text = llama2_wrap_overall_instruction_prompt(prompt)

    with torch.no_grad():
        if args.lora_name == "":
            outputs = model.generate([text], sampling_params)
        else:
            outputs = model.generate([text], sampling_params, lora_request=LoRARequest(sql_lora_path=lora_name))

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

    return generated_text


def obtain_four_type_conf(question, response, prompt_template, tokenizer, model, lora_name):
    step1, step_n_1 = extract_step_answer(response)
    step1 = question + "\n" + step1
    step_n_1 = question + "\n" + step_n_1
    answer = question + "\n" + response
    texts = [question, step1, step_n_1, answer]
    # print(texts)
    # pdb.set_trace()
    conf_answer = []
    for text in texts:
        prompt = prompt_template.replace(" hhx ", text)
        resp = get_response(tokenizer, model, lora_name, prompt, args.model_name)
        conf_answer.append(resp)
    return conf_answer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.lora_name == "":
        model = LLM(model=args.model_path, gpu_memory_utilization=0.7)
    else:
        model = LLM(model=args.model_path, enable_lora=True)
    print("!!!load the model successfully!")

    # Prepare your data
    data = read_json(args.dataPath)
    print(f"=========data path:{args.dataPath}")
    print(f"=========save path:{args.savePath}")

    prompt_template = ""
    if "GSM" in args.savePath:
        res = []
        if args.response_mode == "conf":
            prompt_template = prompt_templates['GSM_conf']
        else:
            prompt_template = prompt_templates['GSM_format']

        for example in tqdm(data):
            if args.response_mode == "conf":
                conf_answer = obtain_four_type_conf(example['question'], example['response'], prompt_template,
                                                    tokenizer, model, args.lora_name)
                example['conf'] = conf_answer
            else:
                prompt = prompt_template.replace(" hhx ", example['question'])
                resp = get_response(tokenizer, model, args.lora_name, prompt)
                example['response'] = resp
            res.append(example)
            save_list_to_json(res, args.savePath)

    elif "CSQA" in args.savePath:
        res = []
        if args.response_mode == "conf":
            prompt_template = prompt_templates['CSQA_conf']
        else:
            prompt_template = prompt_templates['CSQA_format']

        for example in tqdm(data):
            question = example['question']['stem']
            label = ""
            for choice in example['question']['choices']:
                label += choice['label'] + ". " + choice['text'] + "\n"
            label = label.rstrip()
            question = question + "\n" + label

            if args.response_mode == "conf":
                conf_answer = obtain_four_type_conf(question, example['response'], prompt_template, tokenizer, model,
                                                    args.lora_name)
                example['conf'] = conf_answer
            else:
                # pdb.set_trace()
                prompt = prompt_template.replace(" hhx ", f"{question}")
                resp = get_response(tokenizer, model, args.lora_name, prompt)
                example['response'] = resp
            res.append(example)
            save_list_to_json(res, args.savePath)
    elif "Trivia" in args.savePath:
        res = []
        if args.response_mode == "conf":
            prompt_template = prompt_templates['TriviaQA_conf']
        else:
            prompt_template = prompt_templates['TriviaQA_format']

        for example in tqdm(data):
            question = example['question']

            if args.response_mode == "conf":
                conf_answer = obtain_four_type_conf(question, example['response'], prompt_template, tokenizer, model,
                                                    args.lora_name)
                example['conf'] = conf_answer
            else:
                prompt = prompt_template.replace(" hhx ", f"{question}")
                resp = get_response(tokenizer, model, args.lora_name, prompt)
                example['response'] = resp
            res.append(example)
            save_list_to_json(res, args.savePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="/data1/hhx/ckp/conf/qwen7b-qwen7b-full/checkpoint-1000",
                        help="the name of model")
    parser.add_argument("--model_name", type=str, default="qwen",
                        help="the name of model, [qwen, llama2], you can set other models")
    parser.add_argument("--lora_name", type=str, default="")
    parser.add_argument("--dataPath", type=str,
                        default="/data1/hhx/public/confidence/LLMConfidence/methods/ours/data/test/CSQA_result_808.json")
    parser.add_argument("--savePath", type=str,
                        default="/data1/hhx/public/confidence/LLMConfidence/methods/ours/data/test/CSQA_result_808_with_conf.json")
    parser.add_argument("--response_mode", type=str, default="conf", help="[normal, conf]")
    args = parser.parse_args()
    main(args)




