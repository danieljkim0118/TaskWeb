---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5b-pfeiffer-e15-lr1e3-b64-s42
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5b-pfeiffer-e15-lr1e3-b64-s42

This model is a fine-tuned version of [../../../../models/t5-v1.1-lm-base](https://huggingface.co/../../../../models/t5-v1.1-lm-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0223
- Pearson: 79.8528
- Spearmanr: 80.311
- Gen Len: 3.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 64
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 15.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.10.1
- Tokenizers 0.12.1
