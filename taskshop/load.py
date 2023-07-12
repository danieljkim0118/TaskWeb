import json
import numpy as np
import os

def postprocess_metric(metric):
    if isinstance(metric, list):
        assert(len(metric) == 1)
        metric = metric[0]
    if metric > 1.0:
        metric /= 100.0
    return metric

def load_results_dir(task, output_dir, task2metrics):
    output_path = os.path.join(output_dir, "eval_results.json")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as eval_file:
            eval_results = json.load(eval_file)
        result_list = list()
        # Iterate over the pre-specified metrics for a given task
        for k in task2metrics[task]:
            metric_key = f"eval_{k}"
            result_ = eval_results[metric_key]
            result_ = postprocess_metric(result_)
            result_list.append(result_)
        # Average over the metrics
        result = sum(result_list) / len(result_list)
    else:  # the evaluation has not been performed yet!
        print(f"{output_path} does not exist!")
        result = None
        exit(0)
    return result

def load_results(
    target_task, source_task, model, adaptation, epoch, lr, bsz, seed_list, 
    task2dir, task2metrics, baseline=False
):
    target_taskdir = task2dir[target_task]
    taskdir = target_taskdir if baseline else os.path.join(target_taskdir, source_task)
    if adaptation == "finetune":
        model_header = f"{model}-e{epoch}-lr{lr}-b{bsz}"
    else:
        model_header = f"{model}-{adaptation}-e{epoch}-lr{lr}-b{bsz}"
    results = list()
    config_list = [f"{model_header}-s{seed}" for seed in seed_list]
    # Iterate over seeds of the given configuration
    for config_dir in config_list:
        output_dir = os.path.join(taskdir, config_dir)
        result = load_results_dir(target_task, output_dir, task2metrics)     
        results.append(result)
    results = np.array(results)
    return results