target2source_roe = {
    "anli_r1": ["cb", "hellaswag", "rte", "snli", "wsc"],
    "anli_r2": ["cb", "hellaswag", "rte", "snli", "wsc"],
    "anli_r3": ["cb", "hellaswag", "rte", "snli", "wsc"],
    "cb": ["anli_r3", "rte", "snli", "wic", "wsc"],
    "copa": ["cosmosqa", "hellaswag", "socialiqa", "winogrande", "wsc"],
    "hellaswag": ["copa", "cosmosqa", "piqa", "wic", "winogrande"],
    "rte": ["anli_r3", "cb", "mrpc", "qnli", "snli"],
    "story_cloze": ["copa", "cosmosqa", "hellaswag", "socialiqa", "winogrande"],
    "wic": ["mrpc", "piqa", "qqp", "snli", "wsc"],
    "winogrande": ["copa", "hellaswag", "snli", "socialiqa", "wsc"],
    "wsc": ["anli_r3", "cb", "snli", "wic", "winogrande"]
}

target2source_llm = {
    "anli_r1": ["cb", "qnli", "rte", "scitail", "snli"],
    "anli_r2": ["cb", "qnli", "rte", "scitail", "snli"],
    "anli_r3": ["cb", "qnli", "rte", "scitail", "snli"],
    "cb": ["anli_r3", "qnli", "rte", "scitail", "snli"],
    "copa": ["cb", "qnli", "rte", "snli", "socialiqa"],
    "hellaswag": ["cosmosqa", "qnli", "snli", "socialiqa", "winogrande"],
    "rte": ["anli_r3", "cb", "qnli", "scitail", "snli"],
    "story_cloze": ["copa", "hellaswag", "rte", "socialiqa", "winogrande"],
    "wic": ["mrpc", "qnli", "rte", "scitail", "snli"],
    "winogrande": ["copa", "cosmosqa", "hellaswag", "socialiqa", "wsc"],
    "wsc": ["copa", "qnli", "rte", "snli", "winogrande"]
}

target2source_taskshop = {
    "anli_r1": ["cb", "cosmosqa", "rte", "snli", "socialiqa"],
    "anli_r2": ["cb", "cosmosqa", "rte", "snli", "socialiqa"],
    "anli_r3": ["cb", "cosmosqa", "rte", "snli", "socialiqa"],
    "cb": ["anli_r3", "cosmosqa", "snli", "socialiqa", "wsc"],
    "copa": ["cosmosqa", "hellaswag", "piqa", "socialiqa", "winogrande"],
    "hellaswag": ["copa", "cosmosqa", "piqa", "socialiqa", "winogrande"],
    "rte": ["anli_r3", "mrpc", "qnli", "socialiqa", "squad_v2"],
    "story_cloze": ["copa", "cosmosqa", "hellaswag", "socialiqa", "winogrande"],
    "wic": ["anli_r3", "hellaswag", "mrpc", "piqa", "socialiqa"],
    "winogrande": ["copa", "cosmosqa", "piqa", "socialiqa", "wsc"],
    "wsc": ["anli_r3", "rte", "socialiqa", "wic", "winogrande"]
}

target2source_taskweb = {
    "anli_r1": ["boolq", "cosmosqa", "rotten_tomatoes", "rte", "snli"],
    "anli_r2": ["boolq", "cosmosqa", "rotten_tomatoes", "rte", "snli"],
    "anli_r3": ["boolq", "cosmosqa", "rotten_tomatoes", "rte", "snli"],
    "cb": ["anli_r3", "boolq", "rotten_tomatoes", "scitail", "snli"],
    "copa": ["cosmosqa", "piqa", "scitail", "socialiqa", "winogrande"],
    "hellaswag": ["cosmosqa", "piqa", "rotten_tomatoes", "rte", "socialiqa"],
    "rte": ["anli_r3", "cosmosqa", "socialiqa", "squad_v2", "winogrande"],
    "story_cloze": ["cosmosqa", "piqa", "rotten_tomatoes", "socialiqa", "winogrande"],
    "wic": ["anli_r3", "mrpc", "qnli", "rte", "snli"],
    "winogrande": ["anli_r3", "cosmosqa", "quartz", "socialiqa", "squad_v2"],
    "wsc": ["anli_r3", "qnli", "qqp", "snli", "socialiqa"]
}

target2source_all = {
    "anli_r1": ["boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "anli_r2": ["boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "anli_r3": ["boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "cb": ["anli_r3", "boolq", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "copa": ["anli_r3", "boolq", "cb", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "hellaswag": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "rte": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "story_cloze": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande", "wsc"],
    "wic": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "winogrande", "wsc"],
    "winogrande": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "wsc"],
    "wsc": ["anli_r3", "boolq", "cb", "copa", "cosmosqa", "hellaswag", "imdb", "mrpc", "piqa", "qnli", "qqp", "quartz", "rotten_tomatoes", "rte", "scitail", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande"]
}

target2source_top1 = {
    "anli_r1": ["snli"],
    "anli_r2": ["snli"],
    "anli_r3": ["snli"],
    "cb": ["anli_r3"],
    "copa": ["cosmosqa"],
    "hellaswag": ["piqa"],
    "rte": ["anli_r3"],
    "story_cloze": ["cosmosqa"],
    "wic": ["mrpc"],
    "winogrande": ["socialiqa"],
    "wsc": ["anli_r3"]
}

target2source_top3 = {
    "anli_r1": ["cb", "cosmosqa", "snli"],
    "anli_r2": ["cb", "cosmosqa", "snli"],
    "anli_r3": ["cb", "cosmosqa", "snli"],
    "cb": ["anli_r3", "cosmosqa", "snli"],
    "copa": ["cosmosqa", "socialiqa", "winogrande"],
    "hellaswag": ["cosmosqa", "piqa", "socialiqa"],
    "rte": ["anli_r3", "qnli", "squad_v2"],
    "story_cloze": ["copa", "cosmosqa", "hellaswag"],
    "wic": ["anli_r3", "mrpc", "piqa"],
    "winogrande": ["copa", "cosmosqa", "socialiqa"],
    "wsc": ["anli_r3", "snli", "winogrande"]
}

target2source_top10 = {
    "anli_r1": ["cb", "cosmosqa", "hellaswag", "piqa", "qnli", "rte", "scitail", "snli", "socialiqa", "wsc"],
    "anli_r2": ["cb", "cosmosqa", "hellaswag", "piqa", "qnli", "rte", "scitail", "snli", "socialiqa", "wsc"],
    "anli_r3": ["cb", "cosmosqa", "hellaswag", "piqa", "qnli", "rte", "scitail", "snli", "socialiqa", "wsc"],
    "cb": ["anli_r3", "cosmosqa", "hellaswag", "mrpc", "piqa", "qnli", "snli", "socialiqa", "wic", "wsc"],
    "copa": ["anli_r3", "cb", "cosmosqa", "hellaswag", "piqa", "snli", "socialiqa", "wic", "winogrande", "wsc"],
    "hellaswag": ["anli_r3", "copa", "cosmosqa", "piqa", "snli", "socialiqa", "squad_v2", "stsb", "wic", "winogrande"],
    "rte": ["anli_r3", "cb", "cosmosqa", "hellaswag", "mrpc", "qnli", "scitail", "snli", "socialiqa", "squad_v2"],
    "story_cloze": ["anli_r3", "cb", "copa", "cosmosqa", "hellaswag", "piqa", "socialiqa", "wic", "winogrande", "wsc"],
    "wic": ["anli_r3", "cb", "hellaswag", "mrpc", "piqa", "qqp", "scitail", "snli", "socialiqa", "wsc"],
    "winogrande": ["anli_r3", "cb", "copa", "cosmosqa", "hellaswag", "piqa", "snli", "socialiqa", "wic", "wsc"],
    "wsc": ["anli_r3", "cb", "cosmosqa", "hellaswag", "piqa", "rte", "snli", "socialiqa", "wic", "winogrande"]
}

target2source_mix1 = {
    "anli_r3": ["boolq", "hellaswag", "rotten_tomatoes", "rte", "snli"],
    "copa": ["anli_r3", "cosmosqa", "scitail", "socialiqa", "winogrande"],
    "hellaswag": ["cosmosqa", "piqa", "rte", "snli", "socialiqa"],
    "rte": ["anli_r3", "cosmosqa", "snli", "socialiqa", "winogrande"]
}

target2source_mix2 = {
    "anli_r3": ["boolq", "hellaswag", "qqp", "rotten_tomatoes", "snli"],
    "copa": ["anli_r3", "scitail", "snli", "socialiqa", "winogrande"],
    "hellaswag": ["cosmosqa", "qnli", "rte", "snli", "socialiqa"],
    "rte": ["anli_r3", "qqp", "snli", "socialiqa", "winogrande"]
}

target2source_mix3 = {
    "anli_r3": ["copa", "hellaswag", "qqp", "rotten_tomatoes", "snli"],
    "copa": ["anli_r3", "qqp", "scitail", "snli", "socialiqa"],
    "hellaswag": ["cb", "cosmosqa", "qnli", "snli", "socialiqa"],
    "rte": ["mrpc", "qqp", "snli", "socialiqa", "winogrande"]
}

target2source_mix4 = {
    "anli_r3": ["cb", "copa", "hellaswag", "qqp", "rotten_tomatoes"],
    "copa": ["anli_r3", "mrpc", "qqp", "scitail", "snli"],
    "hellaswag": ["cb", "qnli", "qqp", "snli", "socialiqa"],
    "rte": ["copa", "mrpc", "qqp", "snli", "winogrande"]
}

target2source_mix5 = {
    "anli_r3": ["cb", "copa", "hellaswag", "qqp", "quartz"],
    "copa": ["anli_r3", "imdb", "mrpc", "qqp", "snli"],
    "hellaswag": ["cb", "qnli", "qqp", "snli", "squad_v2"],
    "rte": ["copa", "mrpc", "qqp", "snli", "wic"]
}
