# Note

## Written By KYLiN

---

Using datasets : [billsum](https://github.com/tensorflow/datasets/blob/master/docs/catalog/billsum.md)

Full Model Source :
![alt text](images/image.png)

Full Model in TA ROUGE-Lsum : ![alt text](images/image_2.png)

Kaggle Baseline : ![alt text](images/image_kaggle.png)

---

## Pruning Model

Video : [NLP pruning](<https://www.youtube.com/watch?v=UcwDgsMgTu4>)

Torch Pruning : [Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

1. Structured Pruning
![alt text](images/image_structed.png)

2. Knowledge Distillation
![alt text](images/image_knowlege.png)

---

Torch Pruning Blog:

1. [Torch-Pruning | 轻松实现结构化剪枝算法](https://zhuanlan.zhihu.com/p/619482727)

---

## NNI

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

---
T5 Model : [here](https://zhuanlan.zhihu.com/p/647802200)

Pruning : [here](https://github.com/airaria/TextPruner)
