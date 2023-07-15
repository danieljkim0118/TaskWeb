---
language:
- en
tags:
- generated_from_trainer
datasets:
- swag
metrics:
- accuracy
model-index:
- name: roberta-e3-lr1e5-b16-s20
  results:
  - task:
      name: Multiple Choice
      type: multiple-choice
    dataset:
      name: SWAG
      type: swag
      args: regular
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.45593035221099854
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-e3-lr1e5-b16-s20

This model is a fine-tuned version of [../../outputs/qqp/roberta-e3-lr3e5-b8-s20](https://huggingface.co/../../outputs/qqp/roberta-e3-lr3e5-b8-s20) on the SWAG dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6938
- Accuracy: 0.4559

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 32
- seed: 20
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.21.3
- Pytorch 1.10.0+cu113
- Datasets 2.5.1
- Tokenizers 0.12.1
