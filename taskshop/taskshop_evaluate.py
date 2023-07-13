import argparse
import json
import numpy as np
import os
import pickle

from load import load_results
from metrics import ndcg, TASK2METRICS
from utils import combine_scores, get_task2dir, lr2str, taskshop_llm2dir, taskshop_roe2dir, SEED_LIST

PAIRWISE_TRANSFER_DIR = "../data/pairwise_transfer"
SCORE_DIR = "../transfer/outputs_1000"

def load_scores(source_task, target_task, task2dir_transfer, config_dir, model="t5l", adaptation="finetune"):
    """Helper for loading model performances for a given (source -> target) transfer

    Args:
        source_task (str): source task of interest
        target_task (str): target task of interest
        task2dir_transfer (dict[str -> str]): dictionary mapping each task to its directory with transfer scores
        config_dir (str): path to directory with json files containing model hyperparameter configs
        model (str, optional): abbreviation for model of interest. Defaults to "t5l" (T5-large).
        adaptation (str, optional): abbreviation for adaptation method of interest. Defaults to "finetune" (full fine-tuning).        

    Returns:
        float: average model performance across all random seeds for the given (source -> target) transfer
    """
    config_path = os.path.join(config_dir, f"{model}_{adaptation}.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    hyperparam_config_target = config[target_task]
    epoch_tgt = hyperparam_config_target["epochs"]
    lr_tgt = lr2str(hyperparam_config_target["learning_rate"])
    bsz_tgt = hyperparam_config_target["batch_size"]
    if "seed_list" in hyperparam_config_target:
        if source_task in hyperparam_config_target["seed_list"]: 
            seed_list_ = hyperparam_config_target["seed_list"][source_task]
        else:
            seed_list_ = hyperparam_config_target["seed_list"]["default"]
    else:
        seed_list_ = SEED_LIST
    results_target = load_results(
        target_task, source_task, model, adaptation, epoch_tgt, lr_tgt, bsz_tgt, seed_list_, 
        task2dir=task2dir_transfer, task2metrics=TASK2METRICS, baseline=False
    )
    result = np.mean(results_target, axis=0)
    return result

def evaluate_selection(target_task, source_task_list, task2dir_transfer, taskshop_info, transfer_info=None, regret_k=None, method="llm", lmbda=0.7):
    """Evaluate the overall ranking and top-k precision of TaskShop.

    Args:
        target_task (str): target task of interest
        source_task_list (list[str]): list of source tasks
        task2dir_transfer (dict[str -> str]): dictionary mapping each task to its directory with transfer scores
        taskshop_info (dict[str]): dictionary containing information about TaskShop settings
        transfer_info (dict[str], optional): dictionary containing info about the model and adaptation method. Defaults to None.
        regret_k (int, optional): value of k for computing regret @ k (returns NDCG if None). Defaults to None.
        method (str, optional): task selection method. Defaults to "llm".
        lmbda (float, optional): hyperparameter for interpolating pivot-based and direct TaskShop predictions. Defaults to 0.7.

    Returns:
        float: evaluation metric (NDCG or Regret @ k)
    """
    # load TaskShop predictions
    pred_path = taskshop_llm2dir(taskshop_info, transfer_info, lmbda=lmbda) if method == "llm" else taskshop_roe2dir(taskshop_info, transfer_info, lmbda=lmbda)
    with open(pred_path, "rb") as f:
        pred_dict = pickle.load(f)
    # load pairwise transfer scores
    with open(os.path.join(PAIRWISE_TRANSFER_DIR, "pairwise_transfer_scores.p"), "rb") as f:
        label_dict = pickle.load(f)
    source_task_preds = []
    source_task_labels = []
    for source_task in source_task_list:
        score = pred_dict[target_task][source_task]
        source_task_preds.append((source_task, score))
        if transfer_info is None:
            scores = label_dict[(target_task, source_task)][0]
            scores = (scores[0], scores[1])
        else:
            transfer_adapt, transfer_model = transfer_info["adapt"], transfer_info["model"]
            transfer_dict_st = label_dict[(target_task, source_task)][1]
            scores = transfer_dict_st[(transfer_adapt, transfer_model)]
        score_label = combine_scores(*scores)
        source_task_labels.append((source_task, score_label))
    # order TaskShop predictions and actual pairwise transfer scores from most to least similar
    pred_ranking = [task for task, _ in sorted(source_task_preds, key=lambda x: x[1], reverse=True)]
    gold_ranking = [task for task, _ in sorted(source_task_labels, key=lambda x: x[1], reverse=True)]
    assert(len(pred_ranking) == len(gold_ranking))
    assert(sorted(pred_ranking) == sorted(gold_ranking))
    if regret_k:  # compute regret @ k
        # compute average top-5 score according to actual pairwise transfer
        max_label_list =  [load_scores(source_task, target_task, task2dir_transfer) for source_task in gold_ranking[:regret_k]]
        max_label = sum(max_label_list) / len(max_label_list)
        # compute average top-5 score according to TaskShop predictions
        score_list = [load_scores(source_task, target_task, task2dir_transfer) for source_task in pred_ranking[:regret_k]]
        score = sum(score_list) / len(score_list)
        score = (max_label - score) / max_label
    else:  # compute NDCG
        score = ndcg(pred_ranking, gold_ranking)
    return score, pred_ranking, gold_ranking


if __name__ == "__main__":

    # obtain predictions for all tasks in TaskWeb
    task_list = ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"]
    task2dir_transfer = get_task2dir(SCORE_DIR, task_list)

    # allow the user to specify whether they would prefer to use LLM-similarity or RoE (Retrieval-of-Experts)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_selection",
        type=str,
        default="roe",
        help="Task selection method to use (llm or roe)."
    )
    args = parser.parse_args()

    # information regarding LLM-similarity
    llm_info = {
        "model": "text-davinci-003"
    }

    # information regarding RoE
    roe_info = {
        "model": "all-MiniLM-L6-v2",
        "sim_method": "dot",
        "prompt_num": "32"
    }

    # information regarding pairwise transfer
    transfer_info = {
        "adapt": "finetune",
        "model": "t5l"
    }

    # choose task selection method
    taskshop_info = llm_info if args.task_selection == "llm" else roe_info

    ## replicate NDCG and Regret @ k scores reported in the paper
    category2task = {"nli": ["anli_r3", "cb", "qnli", "rte", "snli", "scitail"], "paraphrase": ["mrpc", "qqp", "stsb"], "commonsense": ["copa", "cosmosqa", "hellaswag", "quartz", "socialiqa", "winogrande"], "sentiment": ["imdb", "rotten_tomatoes"], "qa": ["boolq"], "semantics": ["wic", "wsc"]}
    score_avg_all = 0
    score_topk_avg_all = 0
    task_count = 0
    lmbda = 0.7
    # iterate over all task categories
    for category, task_list_custom in category2task.items():
        score_avg = 0
        score_topk_avg = 0
        # iterate over all tasks in each category
        for target_task in task_list_custom:
            source_task_list = [task for task in task_list if task != target_task]
            score, pred_rankings, gold_rankings = evaluate_selection(target_task, source_task_list, task2dir_transfer, taskshop_info, transfer_info, method=args.task_selection, lmbda=lmbda)
            score_topk, pred_rankings, gold_rankings = evaluate_selection(target_task, source_task_list, task2dir_transfer, taskshop_info, transfer_info, regret_k=5, method=args.task_selection, lmbda=lmbda)
            score_avg += score
            score_topk_avg += score_topk
            score_avg_all += score
            score_topk_avg_all += score_topk
            task_count += 1
        score_avg /= len(task_list_custom)
        score_topk_avg /= len(task_list_custom)
        print(category, np.round(100 * score_avg, 2), np.round(100 * score_topk_avg, 2))
    # compute overall score
    score_avg_all /= task_count
    score_topk_avg_all /= task_count
    print(np.round(100 * score_avg_all, 2), np.round(100 * score_topk_avg_all, 2))
