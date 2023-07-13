import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text_input, tokenizer, model):
    encoded_input = tokenizer(text_input, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def similarity_score(source_text, target_text, tokenizer, model, euclidean=True):
    sentence_embeddings_src = encode_text(source_text, tokenizer, model)
    sentence_embeddings_tgt = encode_text(target_text, tokenizer, model)
    if euclidean:
        scores = -1 * torch.cdist(sentence_embeddings_src, sentence_embeddings_tgt, p=2.0)
    else:
        scores = torch.matmul(sentence_embeddings_src, sentence_embeddings_tgt.T)
    return scores