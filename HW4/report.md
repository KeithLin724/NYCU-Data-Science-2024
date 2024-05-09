# 2024 NYCU Data Science HW4 Model Compression Report

## Written By 練鈞揚

---

## Method

In this homework, i using few step to compress the model, I split to a 4 part to compress with different compress ratio, respectively `T5LayerSelfAttention`, `T5LayerCrossAttention`, `T5LayerFF` and `lm_head`, in this homework, i found the `T5LayerCrossAttention` is the key part to decide your score upper bound, And split to few part to compress idea i got it from nni (AutoML) tutorial.

In the second part, I found when the model max length large than 20 , the score will be more better than default. I choose 64 and 128 to be a max length, and the 64 is the best for this homework.

here i provide the sample code how did i compress the model

1. Choose the pruning ratio

    ```python
    self_attention_prune_amount = 0.9
    cross_attention_prune_amount = 0.4
    ff_prune_amount = 0.7
    lm_head_amount = 0.8
    ```

2. Choose the pruning ratio

    ```python
    fine_tune_model_epochs = 15
    self_attention_epochs = 15
    cross_attention_epochs = 20
    ff_prune_epochs = 15
    lm_head_epochs = 30
    ```

3. Split the part

    ```python
    parameters_to_prune = {"self_attention" : [] , "cross_attention":[] , "ffn":[] , "lm_head":[]}
    for name, module in model.named_modules():
        # print(name , type(module))
        if isinstance(module ,T5LayerSelfAttention ):
            # print("SelfAttention " , module)
            
            for name_2 , item in module.named_modules():
                if isinstance(item , torch.nn.Linear):
                    parameters_to_prune["self_attention"].append((item , "weight"))
                    
        if isinstance(module ,T5LayerCrossAttention ):
            # print("CrossAttention " , module)
            
            for name_2 , item in module.named_modules():
                if isinstance(item , torch.nn.Linear):
                    parameters_to_prune["cross_attention"].append((item , "weight"))
                    
        if isinstance(module ,T5LayerFF ):
            # print("FFN " , module)
            
            for name_2 , item in module.named_modules():
                if isinstance(item , torch.nn.Linear):
                    parameters_to_prune["ffn"].append((item , "weight"))
                    
        if isinstance(module ,torch.nn.Linear ) and name == "lm_head":
            parameters_to_prune["lm_head"].append((module , "weight"))
    ```

4. prune the model

    ```python
    trainer.set_train_epochs(fine_tune_model_epochs)

    prune.global_unstructured(
        parameters_to_prune["self_attention"],
        pruning_method=prune.L1Unstructured,
        amount=self_attention_prune_amount,
    )

    trainer.set_train_epochs(self_attention_epochs)
    trainer.train()

    prune.global_unstructured(
        parameters_to_prune["cross_attention"],
        pruning_method=prune.L1Unstructured,
        amount=cross_attention_prune_amount,
    )

    trainer.set_train_epochs(cross_attention_epochs)
    trainer.train()

    prune.global_unstructured(
        parameters_to_prune["ffn"],
        pruning_method=prune.L1Unstructured,
        amount=ff_prune_amount,
    )

    trainer.set_train_epochs(ff_prune_epochs)
    trainer.train()

    prune.global_unstructured(
        parameters_to_prune["lm_head"],
        pruning_method=prune.L1Unstructured,
        amount=lm_head_amount,
    )

    trainer.set_train_epochs(lm_head_epochs)
    trainer.train()
    ```

## Model Structure of T5 small

```txt
T5ForConditionalGeneration(
  (shared): Embedding(32128, 512)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 512)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
              (relative_attention_bias): Embedding(32, 8)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseActDense(
              (wi): Linear(in_features=512, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): ReLU()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-5): 5 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseActDense(
              (wi): Linear(in_features=512, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): ReLU()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): T5LayerNorm()
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): T5Stack(
    (embed_tokens): Embedding(32128, 512)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
              (relative_attention_bias): Embedding(32, 8)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseActDense(
              (wi): Linear(in_features=512, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): ReLU()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-5): 5 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=512, out_features=512, bias=False)
              (k): Linear(in_features=512, out_features=512, bias=False)
              (v): Linear(in_features=512, out_features=512, bias=False)
              (o): Linear(in_features=512, out_features=512, bias=False)
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseActDense(
              (wi): Linear(in_features=512, out_features=2048, bias=False)
              (wo): Linear(in_features=2048, out_features=512, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): ReLU()
            )
            (layer_norm): T5LayerNorm()
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): T5LayerNorm()
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (lm_head): Linear(in_features=512, out_features=32128, bias=False)
)
```

---

## References

### Pytorch official pruning tutorial

Torch Pruning : [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

### NNI

Video : [here](https://www.youtube.com/watch?v=wKh51Jnr0a8)

Github : [here](https://github.com/microsoft/nni/)

Web : [here](https://nni.readthedocs.io/zh/stable/index.html)

Pruning

1. Masking simulation
2. Speedup
3. Fine-tuning

Step

1. Weight matrix
2. ((from NNI Pruner)mask (0, 1) , weight matrix)
3. masked weights
4. NNI Speed up

nni in transformer : [here](https://www.infoq.cn/article/6mA1gDVFWU1oj1ZdQyD2)

nni in transformer example : [here](https://nni.readthedocs.io/en/stable/tutorials/new_pruning_bert_glue.html)

nni config list : [here](https://nni.readthedocs.io/zh/stable/compression/config_list.html)
