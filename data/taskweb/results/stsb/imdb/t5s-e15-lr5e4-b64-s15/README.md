---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5s-e15-lr5e4-b64-s15
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5s-e15-lr5e4-b64-s15

This model is a fine-tuned version of [../../outputs/imdb/t5s-e15-lr5e5-b32-s15](https://huggingface.co/../../outputs/imdb/t5s-e15-lr5e5-b32-s15) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3106
- Pearson: 78.3662
- Spearmanr: 78.3739
- Gen Len: 2.986

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
- seed: 15
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 15.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.10.1
- Tokenizers 0.12.1