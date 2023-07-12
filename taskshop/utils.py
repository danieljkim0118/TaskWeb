import json
import numpy as np
import os
import torch
import torch.nn.functional as F

TASK2FORMAT = {
    "anli_r3": "anli",
    "boolq": "super_glue/boolq",
    "cb": "super_glue/cb",
    "copa": "super_glue/copa",
    "cosmosqa": "cosmos_qa",
    "hellaswag": "hellaswag",
    "imdb": "imdb",
    "mrpc": "glue/mrpc",
    "piqa": "piqa",
    "qnli": "glue/qnli",
    "qqp": "glue/qqp",
    "quartz": "quartz",
    "rotten_tomatoes": "rotten_tomatoes",
    "rte": "super_glue/rte",
    "scitail": "scitail/tsv_format",
    "snli": "snli",
    "socialiqa": "social_i_qa",
    "squad_v2": "squad_v2",
    "story_cloze": "story_cloze/2016",
    "stsb": "glue/stsb",
    "wic": "super_glue/wic",
    "winogrande": "winogrande/winogrande_xl",
    "wsc": "super_glue/wsc.fixed"
}

SEED_LIST=[10, 15, 20, 25, 30, 35, 40, 42]

def combine_scores(score_1, score_2):
    """Our implementation for interpolating the two transfer metrics measuring average percentage change 
    in model performance (PC) and proportion of models with positive transfer (PM).

    Args:
        score_1 (float): average percentage change in model performance (PC).
        score_2 (float): proportion of models with positive transfer (PM).

    Returns:
        float: interpolated pairwise task transfer metric.
    """
    out = 0.5 * score_1 + 0.5 * (0.2 * score_2 - 0.1)
    return out

def format_prompt(prompt):
    prompt = prompt.replace('/', '_')
    prompt = prompt.replace(' ', '_')
    return prompt

def get_answer_index(token_list):
    out = -1
    for i, token in enumerate(token_list):
        if token.strip() == "?" and i >= 3 and token_list[i-1].strip() == "similar" and token_list[i-2].strip() == "these" and token_list[i-3].strip() == "Are":
            out = i + 1
    return out

def get_task2dir(output_dir, task_list):
    task2dir = {task: f"{output_dir}/{task}" for task in task_list}
    return task2dir

def load_data(dataset_path):
    outputs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        input_list = list(f)
    for input_str in input_list:
        data = json.loads(input_str)           
        outputs.append(data)
    return outputs

def save_data(outputs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for output in outputs:
            json.dump(output, f)
            f.write('\n')

def llm2dir(llm_info):
    model = llm_info["model"]
    instructions = llm_info["instructions"]
    path = os.path.join("scores", instructions, f"scores_{model}.p")
    return path

def retriever2dir(retriever_info):
    model = retriever_info["model"]
    sim_method = retriever_info["sim_method"]
    prompt_num = retriever_info["prompt_num"]
    path = os.path.join("scores", "roe", f"scores_{model}_{sim_method}_{prompt_num}.p")
    return path

def taskshop_llm2dir(taskshop_info, transfer_info, lmbda):
    model = taskshop_info["model"]
    if transfer_info is None:
        path = os.path.join("scores", "taskshop", f"all_llm_{model}_lmbda0{int(lmbda * 10)}.p")
    else:
        path = os.path.join("scores", "taskshop", f"{transfer_info['model']}_{transfer_info['adapt']}_llm_{model}_lmbda0{int(lmbda * 10)}.p")
    return path

def taskshop_retriever2dir(taskshop_info, transfer_info, lmbda):
    model = taskshop_info["model"]
    sim_method = taskshop_info["sim_method"]
    prompt_num = taskshop_info["prompt_num"]
    if transfer_info is None:
        path = os.path.join("scores", "taskshop", f"all_retriever_{model}_{sim_method}_{prompt_num}_lmbda0{int(lmbda * 10)}.p")
    else:
        path = os.path.join("scores", "taskshop", f"{transfer_info['model']}_{transfer_info['adapt']}_retriever_{model}_{sim_method}_{prompt_num}_lmbda0{int(lmbda * 10)}.p")
    return path

def lr2str(learning_rate):
    out = str("{:.0e}".format(learning_rate))
    return f"{out[0]}e{out[-1]}"
