def accuracy(predictions, references):
    comp = [p == r for p, r in zip(predictions, references)]
    return sum(comp) / len(comp)

def tp_tn_fp_fn(predictions, references, pos_label):
    tp = len([0 for p, r in zip(predictions, references) if p == pos_label and r == pos_label])
    fp = len([0 for p in predictions if p == pos_label]) - tp
    fn = len([0 for r in references if r == pos_label]) - tp
    return tp, fp, fn

def f1_binary(predictions, references, pos_label):
    tp, fp, fn = tp_tn_fp_fn(predictions, references, pos_label)
    precision_pos = tp / (tp + fp)
    recall_pos = tp / (tp + fn)
    if precision_pos + recall_pos > 0:
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
        return f1_pos
    else:
        return .0

def f1_score(predictions, references, pos_label, label_list, average="binary"):
    f1_bin = f1_binary(predictions, references, pos_label)
    if average == "binary":
        return f1_bin
    label2f1 = dict()
    label2f1[pos_label] = f1_bin
    for label in label_list:
        if label != pos_label:
             f1 = f1_binary(predictions, references, label)
             label2f1[label] = f1
    if average == "macro":
        f1_list = sorted(list(label2f1.values()))
        return sum(f1_list) / len(f1_list)
    elif average == "micro":  # if there is a single class for each label, then micro F1 = accuracy
        return accuracy(predictions, references)
    elif average == "weighted":
        f1_weighted = .0
        for label, f1 in label2f1.items():
            support = len([0 for r in references if r == label])
            f1_weighted += f1 * support
        f1_weighted /= len(references)
        return f1_weighted
    else:
        raise NotImplementedError("Please specify the appropriate average value!")