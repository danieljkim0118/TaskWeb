---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5l-pfeiffer-e15-lr5e5-b32-s15
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5l-pfeiffer-e15-lr5e5-b32-s15

This model is a fine-tuned version of [../../../../models/t5-v1.1-lm-large](https://huggingface.co/../../../../models/t5-v1.1-lm-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9978
- Pearson: 87.3515
- Spearmanr: 87.7002
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
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 15
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- num_epochs: 15.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.10.1
- Tokenizers 0.12.1
