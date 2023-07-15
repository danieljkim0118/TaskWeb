---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5l-e5-lr1e4-b32-s20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5l-e5-lr1e4-b32-s20

This model is a fine-tuned version of [../../outputs/cosmosqa/t5l-e5-lr5e5-b16-s20](https://huggingface.co/../../outputs/cosmosqa/t5l-e5-lr5e5-b16-s20) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9470
- Pearson: 86.4424
- Spearmanr: 86.4777
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
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 32
- seed: 20
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.5.1
- Tokenizers 0.12.1
