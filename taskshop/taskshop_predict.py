import argparse
import numpy as np
import os
import pickle

from tqdm import tqdm
from utils import combine_scores, llm2dir, retriever2dir

PAIRWISE_TRANSFER_DIR = "../data/pairwise_transfer"

def process_scores(score_list):
    """post-process pairs of scores from source -> pivot and pivot -> target transfers.

    Args:
        score_list (list[(str, str, str)]): list of triples containing (pivot task, source -> pivot and pivot -> target scores)

    Returns:
        float: score representing the source -> pivot -> target transfer.
    """
    score_avg_max = -10
    score_avg_mean = 0
    for _, score_1, score_2 in score_list:
        score_avg = 0.5 * score_1 + 0.5 * score_2
        score_avg_max = max(score_avg_max, score_avg)
        score_avg_mean += score_avg
    score_avg_mean /= len(score_list)
    out = score_avg_mean
    return out

def predict_transfer_llm(target_task, source_task, pivot_task_list, transfer_info, llm_info, lmbda):
    """Implementation of TaskShop using LLM-similarity scores for a given (source, target) pair.

    Args:
        target_task (str): target task of interest
        source_task (str): source task of interest
        pivot_task_list (list[str]): list of pivot tasks to be used for prediction
        transfer_info (dict[str]): dictionary containing info about the model and adaptation method
        llm_info (dict[str]): dictionary containing info about the LLM used to assign inter-task similarity scores
        lmbda (float): hyperparameter for interpolating pivot-based and direct TaskShop predictions

    Returns:
        float: TaskShop prediction for the (source, target) pair
    """
    # load LLM-similarity scores
    with open(llm2dir(llm_info), "rb") as f:
        pred_dict = pickle.load(f)
    # load pairwise transfer scores
    with open(os.path.join(PAIRWISE_TRANSFER_DIR, "pairwise_transfer_scores.p"), "rb") as f:
        label_dict = pickle.load(f)
    transfer_adapt, transfer_model = transfer_info["adapt"], transfer_info["model"]
    # direct score is averaged between source -> target and target -> source prediction since LLM predictions are direction-agnostic
    score_direct = (pred_dict[(source_task, target_task)][0] + pred_dict[(target_task, source_task)][0]) / 2
    pivottask2score = {}
    # compute transfer from source to each pivot task
    for pivot_task in pivot_task_list:
        if pivot_task != source_task and pivot_task != target_task:
            if transfer_info is None:
                scores = label_dict[(pivot_task, source_task)][0]
                scores = (scores[0], scores[1])
            else:
                transfer_adapt, transfer_model = transfer_info["adapt"], transfer_info["model"]
                transfer_dict_st = label_dict[(pivot_task, source_task)][1]
                scores = transfer_dict_st[(transfer_adapt, transfer_model)]
            score_label = combine_scores(*scores)
            pivottask2score[pivot_task] = score_label
    score_list = []
    ## Note: LLM-based probability scores need re-normalization to be intergrated with pairwise transfer scores
    # compute statistics to re-normalize pivot-based score
    score_2_mean = np.mean([x[0] for x in list(pred_dict.values())])
    score_2_std = np.std([x[0] for x in list(pred_dict.values())])
    # compute scores over all pivot tasks
    for pivot_task, score_1 in pivottask2score.items():
        score_2 = (pred_dict[(pivot_task, target_task)][0] + pred_dict[(target_task, pivot_task)][0]) / 2
        score_2 = (score_2 - score_2_mean) / score_2_std
        score_list.append((pivot_task, score_1, score_2))
    score_pivot = process_scores(score_list, target_task, source_task)
    # compute statistics to re-normalize direct score
    score_2_max = max([x[0] for x in list(pred_dict.values())])
    score_2_min = min([x[0] for x in list(pred_dict.values())])
    score_direct = (score_direct - score_2_min) / (score_2_max - score_2_min)
    # interpolate scores between pivot-based and direct estimate
    score = lmbda * score_pivot + (1 - lmbda) * score_direct
    return score

def predict_transfer_roe(target_task, source_task, pivot_task_list, transfer_info, retriever_info, lmbda):
    """Implementation of TaskShop using Retrieval-of-Experts (RoE) scores for a given (source, target) pair.

    Args:
        target_task (str): target task of interest
        source_task (str): source task of interest
        pivot_task_list (list[str]): list of pivot tasks to be used for prediction
        transfer_info (dict[str]): dictionary containing info about the model and adaptation method
        retriever_info (dict[str]): dictionary containing info about RoE used to assign inter-task similarity scores
        lmbda (float): hyperparameter for interpolating pivot-based and direct TaskShop predictions

    Returns:
        float: TaskShop prediction for the (source, target) pair
    """
    # load Retrieval-of-Experts scores
    with open(retriever2dir(retriever_info), "rb") as f:
        pred_dict = pickle.load(f)
    # load pairwise transfer scores
    with open(os.path.join(PAIRWISE_TRANSFER_DIR, "pairwise_transfer_scores.p"), "rb") as f:
        label_dict = pickle.load(f)
    if transfer_info is not None:
        transfer_adapt, transfer_model = transfer_info["adapt"], transfer_info["model"]
    score_direct = pred_dict[target_task][source_task]
    pivottask2score = {}
    # compute transfer from source to each pivot task
    for pivot_task in pivot_task_list:
        if pivot_task != source_task and pivot_task != target_task:
            if transfer_info is None:
                scores = label_dict[(pivot_task, source_task)][0]
                scores = (scores[0], scores[1])
            else:
                transfer_dict_st = label_dict[(pivot_task, source_task)][1]
                scores = transfer_dict_st[(transfer_adapt, transfer_model)]
            score_label = combine_scores(*scores)
            pivottask2score[pivot_task] = score_label
    score_list = []
    # compute scores over all pivot tasks
    for pivot_task, score_1 in pivottask2score.items():
        score_2 = pred_dict[target_task][pivot_task]
        score_list.append((pivot_task, score_1, score_2))
    score_pivot = process_scores(score_list, target_task, source_task)
    # interpolate scores between pivot-based and direct estimate
    score = lmbda * score_pivot + (1 - lmbda) * score_direct
    return score

def predict_transfer_taskshop(target_task_list, source_task_list, transfer_info, taskshop_info, method="roe", pivot_task_list=None, lmbda=0.7, output_dir="scores/taskshop"):
    """Implementation of TaskShop.

    Args:
        target_task_list (list[str]): list of target tasks of interest
        source_task_list (list[str]): list of source tasks of interest
        transfer_info (dict[str]): dictionary containing info about the model and adaptation method
        taskshop_info (dict[str]): dictionary containing info about LLM-similarity/RoE used to assign inter-task similarity scores
        method (str, optional): task selection method. Defaults to "roe".
        pivot_task_list (list[str], optional): list of predefined pivot tasks. Defaults to None.
        lmbda (float, optional): hyperparameter for interpolating pivot-based and direct TaskShop predictions. Defaults to 0.7.
        output_dir (str, optional): output directory for storing TaskShop scores. Defaults to "scores/taskshop".

    Raises:
        ValueError: if the task selection method does not fall into one of the supported methods (llm, roe)

    Returns:
        float: TaskShop prediction for the (source, target) pair
    """
    output = {}
    # iterate over all target tasks
    for target_task in tqdm(target_task_list):
        output[target_task] = {}
        # iterate over all source tasks for each target
        for source_task in source_task_list:
            if target_task != source_task:
                # use a set of predefined pivot tasks
                if pivot_task_list is None:
                    pivot_task_list_ = [task for task in source_task_list if task != source_task and task != target_task and task != "squad_v2"]
                else:
                    pivot_task_list_ = pivot_task_list
                # obtain score prediction based on the user-provided method
                if method == "llm":
                    score = predict_transfer_llm(target_task, source_task, pivot_task_list_, transfer_info, taskshop_info, lmbda)
                elif method == "roe":
                    score = predict_transfer_roe(target_task, source_task, pivot_task_list_, transfer_info, taskshop_info, lmbda)
                else:
                    raise ValueError("Please specify a method between 'llm' and 'retriever'!")
                # rescale score for easier analysis
                score = 5 * score - 0.6
                output[target_task][source_task] = score
    # format output path for storing TaskShop scores
    if transfer_info is None:
        if method == "llm":
            output_path = f"{output_dir}/all_{method}_{taskshop_info['model']}.p"
        else:
            output_path = f"{output_dir}/all_{method}_{taskshop_info['model']}_{taskshop_info['sim_method']}_{taskshop_info['prompt_num']}.p"
    else:
        transfer_adapt, transfer_model = transfer_info["adapt"], transfer_info["model"]
        if method == "llm":
            output_path = f"{output_dir}/{transfer_model}_{transfer_adapt}_{method}_{taskshop_info['model']}_lmbda0{int(lmbda * 10)}.p"
        else:
            output_path = f"{output_dir}/{transfer_model}_{transfer_adapt}_{method}_{taskshop_info['model']}_{taskshop_info['sim_method']}_{taskshop_info['prompt_num']}_lmbda0{int(lmbda * 10)}.p"
    # save the scores
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    return output


if __name__ == "__main__":

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
        "model": "text-davinci-003",
        "instructions": "natinstruct"
    }

    # information regarding RoE
    roe_info = {
        "model": "all-MiniLM-L6-v2",
        "sim_method": "dot",
        "prompt_num": "32"
    }

    # information regarding pairwise transfer scores
    transfer_info = {
        "adapt": "finetune",
        "model": "t5l"
    }

    # choose task selection method
    taskshop_info = llm_info if args.task_selection == "llm" else roe_info

    # list all source tasks
    source_task_list = ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"]

    # list all target tasks
    target_task_list = [task for task in source_task_list if task != "squad_v2"] + ["story_cloze"]

    # list all random seeds used for pairwise task transfer
    seed_list = [10, 15, 20, 25, 30, 35, 40, 42]

    # lambda was chosen based on holding out PIQA as validation task
    predict_transfer_taskshop(target_task_list, source_task_list, transfer_info, taskshop_info, method=args.task_selection, pivot_task_list=None, lmbda=0.7)
