import os
import pickle
import random
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from encode import encode_text
from utils import TASK2FORMAT

DATA_DIR = "../data/prompts"

class Task2Prompt:

    def __init__(self, tokenizer, model, num, seed=42):
        self.num = num
        self.seed = seed
        random.seed(seed)
        self.task2format = TASK2FORMAT
        self.tokenizer = tokenizer
        self.model = model
        self.prompt_path = os.path.join(DATA_DIR, f"prompt_info_{num}.p")
        with open(self.prompt_path, "rb") as f:
            self.prompt_data = pickle.load(f)

    def task2embeddings(self, task_name):
        prompt_data = self.prompt_data[task_name]
        prompt_list = [f"{data['choices']}</s>{data['input']}" for data in prompt_data]  # Jang et al., 2023
        embeddings = encode_text(prompt_list, self.tokenizer, self.model)
        return embeddings
    
    def save_embeddings(self, task_list, model_name):
        task_embeddings = {task_name: self.task2embeddings(task_name) for task_name in tqdm(task_list)}
        with open(f"embeddings/task_embeddings_{model_name}_{self.num}_new.p", "wb") as f:
            pickle.dump(task_embeddings, f)

def compute_similarity(target_task, source_task_list, model_name, num_src=100, num_tgt=32, method="euclid"):
    """Compute similarity between prompt embeddings from target task and a list of source tasks on example-by-example level

    Args:
        target_task (str): target task of interest
        source_task_list (list[str]): list of source tasks of interest
        model_name (str): abbreviation for model of interest
        num_src (int, optional): number of examples from the source task. Defaults to 100.
        num_tgt (int, optional): number of examples from the target task. Defaults to 32.
        method (str, optional): math operation for measuring embedding similarity. Defaults to "euclid".

    Returns:
        list[Array[num_tgt x num_src]]: a list over all source tasks containing similarity scores between target examples and source examples
    """
    with open(f"embeddings/task_embeddings_{model_name}_{num_tgt}.p", "rb") as f:
        task_embeddings_tgt = pickle.load(f)
    with open(f"embeddings/task_embeddings_{model_name}_{num_src}.p", "rb") as f:
        task_embeddings_src = pickle.load(f)
    target_embeddings = task_embeddings_tgt[target_task]
    scores_list = []
    for source_task in source_task_list:  # loop inevitable due to different number of prompts for each source task
        source_embeddings = task_embeddings_src[source_task]
        if method == "euclid":
            scores = -1 * torch.cdist(target_embeddings, source_embeddings)
        else:
            scores = torch.matmul(target_embeddings, source_embeddings.T)
        scores_list.append(scores)
    return scores_list

def compute_scores(target_task, source_task_list, model_name, num_src=100, num_tgt=32, method="euclid"):
    """Compute similarity between prompt embeddings from target task and a list of source tasks

    Args:
        target_task (str): target task of interest
        source_task_list (list[str]): list of source tasks of interest
        model_name (str): abbreviation for model of interest
        num_src (int, optional): number of examples from the source task. Defaults to 100.
        num_tgt (int, optional): number of examples from the target task. Defaults to 32.
        method (str, optional): math operation for measuring embedding similarity. Defaults to "euclid".

    Returns:
        dict[str -> float]: dictionary containing similarity scores between each source task in the given list and the target task
    """
    # score dimensions: num_sources x num_target_examples x num_source_examples
    scores_list = compute_similarity(target_task, source_task_list, model_name=model_name, num_src=num_src, num_tgt=num_tgt, method=method)
    # identify the maximally similar source example for each target example for each source task
    max_sim_list = [torch.max(scores, dim=-1)[0] for scores in scores_list]
    max_sim_2d = torch.stack(max_sim_list, dim=-1)
    sim_scores = torch.mean(max_sim_2d, dim=0)
    score_dict = {source_task_list[i]: sim_score.item() for i, sim_score in enumerate(sim_scores)}
    return score_dict

def compute_scores_all(task_list, model_name, num_src=100, num_tgt=32, method="dot"):
    """Compute similarity between prompt embeddings from target task and a list of source tasks

    Args:
        task_list (list[str]): list of tasks of interest
        model_name (str): abbreviation for model of interest
        num_src (int, optional): number of examples from the source task. Defaults to 100.
        num_tgt (int, optional): number of examples from the target task. Defaults to 32.
        method (str, optional): math operation for measuring embedding similarity. Defaults to "euclid".

    Returns:
        dict[str -> dict[str -> float]]: dictionary of target tasks mapping to dictionaries of source tasks with their associated similarity scores
    """
    score_dict_all = {}
    for target_task in tqdm(task_list):
        source_task_list = [task for task in task_list if task != target_task]
        score_dict_target = compute_scores(target_task, source_task_list, model_name=model_name, num_src=num_src, num_tgt=num_tgt, method=method)
        score_dict_all[target_task] = score_dict_target
    with open(f"scores/roe/scores_{model_name}_{method}_{num_tgt}.p", "wb") as f:
        pickle.dump(score_dict_all, f)
    return score_dict_all


if __name__ == "__main__":

    # load tokenizer and model from HuggingFace Hub
    model_path = "all-MiniLM-L6-v2"
    model_name = model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # list of all tasks
    task_list = ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "story_cloze", "stsb", "wic", "winogrande", "wsc"]

    # Step 1: compute and save embeddings
    prompter = Task2Prompt(tokenizer, model, num=100)
    prompter.save_embeddings(task_list, model_name)

    # Step 2: use the embeddings to compute similarity scores between tasks
    score_dict_all = compute_scores_all(task_list, model_name=model_name, num_src=100, num_tgt=32, method='dot')
