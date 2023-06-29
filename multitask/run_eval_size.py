from eval_utils import process_results, load_score_t5
from multitask_info import target2source_all, target2source_top1, target2source_top3, target2source_top10, target2source_taskshop

if __name__ == "__main__":

    # list of all target tasks and random seeds
    task_list = ["anli_r1", "anli_r2", "anli_r3", "cb", "copa", "hellaswag", "rte", "story_cloze", "wic", "winogrande", "wsc"]
    seed_list = [15, 30, 42]

    # T5-3B zero-shot evaluation results with top-1 task
    process_results(task_list, load_score_t5, multitask_dict=target2source_top1, result_name="T5-XL top-1", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with top-3 task
    process_results(task_list, load_score_t5, multitask_dict=target2source_top3, result_name="T5-XL top-3", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with top-5 task
    process_results(task_list, load_score_t5, multitask_dict=target2source_taskshop, result_name="T5-XL top-5", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with top-10 task
    process_results(task_list, load_score_t5, multitask_dict=target2source_top10, result_name="T5-XL top-10", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with top-21 tasks (all tasks except target task)
    process_results(task_list, load_score_t5, multitask_dict=target2source_all, result_name="T5-XL top-21", seed_list=seed_list)
