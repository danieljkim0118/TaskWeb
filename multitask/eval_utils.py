import json
import numpy as np
import os

from utils import format_prompt, hyperparam2str, list2str

TASK2METRICS = {
    "anli_r1": ["accuracy"],
    "anli_r2": ["accuracy"],
    "anli_r3": ["accuracy"],
    "boolq": ["accuracy"],
    "cb": ["accuracy"],
    "copa": ["accuracy"],
    "cosmosqa": ["accuracy"],
    "hellaswag": ["accuracy"],
    "imdb": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "piqa": ["accuracy"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "quartz": ["accuracy"],
    "rotten_tomatoes": ["accuracy"],
    "rte": ["accuracy"],
    "scitail": ["accuracy"],
    "snli": ["accuracy"],
    "socialiqa": ["accuracy"],
    "squad_v2": ["exact", "f1"],
    "story_cloze": ["accuracy"],
    "stsb": ["pearson", "spearmanr"],
    "wic": ["accuracy"],
    "winogrande": ["accuracy"],
    "wsc": ["accuracy"]
}

TASK2PROMPT = {
    "anli_r1": [
      "based on the previous passage",
      "can we infer",
      "does this imply",
      "must be true",
      "should assume"
    ],
    "anli_r2": [
      "based on the previous passage",
      "can we infer",
      "does this imply",
      "must be true",
      "should assume"
    ],
    "anli_r3": [
      "based on the previous passage",
      "can we infer",
      "does this imply",
      "must be true",
      "should assume"
    ],
    "cb": [
      "based on the previous passage",
      "can we infer",
      "does it follow that",
      "does this imply",
      "must be true",
      "should assume"
    ],
    "copa": [
      "cause_effect",
      "choose",
      "i_am_hesitating",
      "more likely"
    ],
    "hellaswag": [
      "complete_first_then",
      "Predict ending with hint",
      "Randomized prompts template"
    ],
    "rte": [
      "based on the previous passage",
      "can we infer",
      "does this imply",
      "must be true",
      "should assume"
    ],
    "story_cloze": [
      "Answer Given options",
      "Choose Story Ending",
      "Movie What Happens Next",
      "Novel Correct Ending",
      "Story Continuation and Options"
    ],
    "wic": [
      "GPT-3-prompt",
      "GPT-3-prompt-with-label",
      "affirmation_true_or_false",
      "question-context",
      "question-context-meaning"
    ],
    "winogrande": [
      "Replace",
      "does underscore refer to",
      "stand for",
      "underscore refer to"
    ],
    "wsc": [
      "p is/are r",
      "the pronoun refers to",
      "in other words"
    ]
}

def compute_scores(input_dir, model_dir_list, target_task):
    eval_path_list = [os.path.join(input_dir, model_dir, "eval_results.json") for model_dir in model_dir_list]
    scores = []
    for eval_path in eval_path_list:
        with open(eval_path, "r", encoding="utf-8") as eval_file:
            eval_data = json.load(eval_file)
        eval_metrics = TASK2METRICS[target_task]
        eval_scores = [eval_data[f"eval_{metric}"] for metric in eval_metrics]
        eval_scores = [s / 100 if s > 1 else s for s in eval_scores]
        eval_score = np.mean(eval_scores, axis=0)
        scores.append(np.round(eval_score, 4))
    return scores

def load_score_t0(target_task, model_name="t03b", prompt_name=None):
    input_dir = os.path.join(f"baseline-zs", target_task)
    model_dir_list = [f"{model_name}-{prompt_name}"]
    scores = compute_scores(input_dir, model_dir_list, target_task=target_task)
    return scores

def load_score_t5(target_task, multitask_list, model_name, prompt_name=None, sample_size=2000, lr=1e-4, seed_list=None):
    result_dir = f"results_uniform{sample_size}"
    config = {"source_learning_rate": lr}
    if prompt_name is None:
        input_dir = os.path.join(result_dir, target_task, list2str(multitask_list))
    else:
        input_dir = os.path.join(result_dir, target_task, "prompt-zs", list2str(multitask_list))
    model_dir_list = hyperparam2str(model_name, prompt_name, config, seed_list=seed_list)
    scores = compute_scores(input_dir, model_dir_list, target_task=target_task)
    return scores

def process_results(task_list, load_function, multitask_dict, result_name, model_name="t5xl", seed_list=None, sample_size=2000, task2prompt=TASK2PROMPT):
    score_avg = 0
    score_list = []
    print(f"========== Computing scores for {result_name} ==========")
    for target_task in task_list:
        prompt_list = task2prompt[target_task]
        t5_score_task = 0
        t5_scores_task = []
        for prompt_name in prompt_list:
            if seed_list is None and model_name == "t03b":
                prompt_name = format_prompt(prompt_name)
                t5_scores_prompt = load_function(target_task=target_task, model_name=model_name, prompt_name=prompt_name)
                t5_score_prompt = np.round(100 * np.mean(t5_scores_prompt), 2)
            else:
                prompt_name = format_prompt(prompt_name)
                multitask_list = multitask_dict[target_task]
                t5_scores_prompt = load_function(target_task=target_task, multitask_list=multitask_list, model_name=model_name, prompt_name=prompt_name, seed_list=seed_list, sample_size=sample_size)
                t5_score_prompt = np.round(100 * np.mean(t5_scores_prompt), 2)
            t5_score_task += t5_score_prompt
            t5_scores_task.append(t5_score_prompt)
        t5_score_task /= len(prompt_list)
        t5_score_task = np.round(t5_score_task, 2)
        score_list.append(t5_score_task)
        print(f"{target_task}: ", t5_score_task, t5_scores_task)
        score_avg += t5_score_task
    score_avg /= len(task_list)
    score_avg = np.round(score_avg, 2)
    print("Average score: ", score_avg)
    return score_avg, score_list
