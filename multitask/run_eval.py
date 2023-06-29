from eval_utils import process_results, load_score_t5
from multitask_info import target2source_all, target2source_roe, target2source_llm, target2source_taskshop, target2source_taskweb

if __name__ == "__main__":

    # list of all target tasks and random seeds
    task_list = ["anli_r1", "anli_r2", "anli_r3", "cb", "copa", "hellaswag", "rte", "story_cloze", "wic", "winogrande", "wsc"]
    seed_list = [15, 30, 42]

    # T5-3B zero-shot evaluation results with all tasks except target task
    process_results(task_list, load_score_t5, multitask_dict=target2source_all, result_name="T5-XL all", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with Retrieval-of-Experts (RoE)
    process_results(task_list, load_score_t5, multitask_dict=target2source_roe, result_name="T5-XL RoE", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with LLM similarity scores (LLM)
    process_results(task_list ,load_score_t5 , multitask_dict=target2source_llm, result_name="T5-XL LLM", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with TaskShop
    process_results(task_list, load_score_t5, multitask_dict=target2source_taskshop, result_name="T5-XL TaskShop", seed_list=seed_list)

    # T5-3B zero-shot evaluation results with TaskWeb
    process_results(task_list, load_score_t5, multitask_dict=target2source_taskweb, result_name="T5-XL TaskWeb", seed_list=seed_list)
