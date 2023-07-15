---
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: t5s-e15-lr5e4-b64-s42
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5s-e15-lr5e4-b64-s42

This model is a fine-tuned version of [../../outputs/wic/t5s-e15-lr5e5-b64-s42](https://huggingface.co/../../outputs/wic/t5s-e15-lr5e5-b64-s42) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3224
- Pearson: 81.8042
- Spearmanr: 81.2919
- Gen Len: 2.9967

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