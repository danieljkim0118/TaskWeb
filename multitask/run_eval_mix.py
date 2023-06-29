from eval_utils import process_results, load_score_t5
from multitask_info import target2source_taskweb, target2source_mix1, target2source_mix2, target2source_mix3, target2source_mix4, target2source_mix5

if __name__ == "__main__":

    # list of all target tasks and random seeds
    task_list = ["anli_r3", "copa", "hellaswag", "rte"]
    seed_list = [15, 30, 42]

    # T5-3B zero-shot evaluation results with TaskWeb (top-5)
    process_results(task_list, load_score_t5, multitask_dict=target2source_taskweb, result_name="T5-XL original", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with 1 unhelpful source task mixed
    process_results(task_list, load_score_t5, multitask_dict=target2source_mix1, result_name="T5-XL mix-1", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with 2 unhelpful source tasks mixed
    process_results(task_list, load_score_t5, multitask_dict=target2source_mix2, result_name="T5-XL mix-2", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with 3 unhelpful source tasks mixed
    process_results(task_list, load_score_t5, multitask_dict=target2source_mix3, result_name="T5-XL mix-3", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with 4 unhelpful source tasks mixed
    process_results(task_list, load_score_t5, multitask_dict=target2source_mix4, result_name="T5-XL mix-4", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with 5 unhelpful source tasks mixed
    process_results(task_list, load_score_t5, multitask_dict=target2source_mix5, result_name="T5-XL mix-5", seed_list=seed_list)
