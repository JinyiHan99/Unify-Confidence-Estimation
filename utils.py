import json
import pickle

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def read_json_line(path):
    res = []
    with open(path,"r") as file:
        for line in file:
            data = json.loads(line.strip())
            res.append(data)
    return res

def save_list_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def qwen_wrap_overall_instruction_prompt(tokenizer, prompt):
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
    return text


def llama2_wrap_overall_instruction_prompt(prompt):
    overall_instruction = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    text = f"<s>{B_INST} {B_SYS} {overall_instruction} {E_SYS} {prompt} {E_INST} "
    return text
            
