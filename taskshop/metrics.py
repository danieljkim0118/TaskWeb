import numpy as np

TASK2METRICS = {
    "anli_r3": ["accuracy"],
    "boolq": ["accuracy"],
    "cb": ["accuracy", "f1"],
    "copa": ["accuracy"],
    "cosmosqa": ["accuracy"],
    "hellaswag": ["accuracy"],
    "hellaswag_old": ["accuracy"],
    "imdb": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    "piqa": ["accuracy"],
    "piqa_old": ["accuracy"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "quartz": ["accuracy"],
    "rotten_tomatoes": ["accuracy"],
    "rte": ["accuracy"],
    "scitail": ["accuracy"],
    "snli": ["accuracy"],
    "socialiqa": ["accuracy"],
    "squad_v2": ["exact", "f1"],
    "stsb": ["pearson", "spearmanr"],
    "wic": ["accuracy"],
    "winogrande": ["accuracy"],
    "winogrande_old": ["accuracy"],
    "wsc": ["accuracy"]
}

def dcg(pred_ranking, gold_ranking):
    """Compute the DCG of a predicted ranking w.r.t. gold ranking

    Args:
        pred_ranking (list[any]): predicted list of objects in ranking order.
        gold_ranking (list[any]): gold list of objects in ranking order.

    Returns:
        float: DCG of predicted ranking
    """
    gold2rank = dict()
    for i, gold in enumerate(gold_ranking):
        gold2rank[gold] = i
    ranking = [gold2rank[pred] for pred in pred_ranking]
    relevance = np.array([len(gold_ranking) - r for r in ranking])
    gains = 2 ** relevance - 1
    discounts = np.log2(np.arange(len(gold_ranking)) + 2)
    return np.sum(gains / discounts, axis=0)

def ndcg(pred_ranking, gold_ranking):
    """Compute the NDCG of a predicted ranking w.r.t. gold ranking

    Args:
        pred_ranking (list[any]): predicted list of objects in ranking order.
        gold_ranking (list[any]): gold list of objects in ranking order.

    Returns:
        float: NDCG of predicted ranking
    """
    return dcg(pred_ranking, gold_ranking) / dcg(gold_ranking, gold_ranking)