---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5b-e10-lr5e4-b64-s25
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5b-e10-lr5e4-b64-s25

This model is a fine-tuned version of [../../outputs/cosmosqa/t5b-e10-lr5e5-b32-s25](https://huggingface.co/../../outputs/cosmosqa/t5b-e10-lr5e5-b32-s25) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2234
- Pearson: 85.0796
- Spearmanr: 84.9055
- Gen Len: 2.9933

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 64
- eval_batch_size: 64
- seed: 25
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.5.1
- Tokenizers 0.12.1
