# Neural Relation Extraction for Chinese

Bi-directional GRU with Word and Sentence Dual Attentions for Relation Extraction

Original Code in https://github.com/thunlp/TensorFlow-NRE, modified for Chinese.


### Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

* scikit-learn (>=0.18)


### Usage:

* Training:

1. Prepare data in origin_data/ , including relation types (relation2id.txt), training data (train.txt), testing data (test.txt) and Chinese word vectors (vec.txt).

2. Prepare data
```
python initial.py
```

3. Training
```
python train_GRU.py
```

* Inference:

```
python test_GRU.py
```

Program will ask for data input in the format of "name1 name2 sentence".


### Sample Results:

```
INFO:tensorflow:Restoring parameters from ./model/ATT_GRU_model-9000
reading word embedding data...
reading relation to id

实体1: 李晓华
实体2: 王大牛
李晓华和王大牛是一对甜蜜的夫妻，前日他们一起去英国旅行。
关系是:
No.1: 夫妻, Probability is 0.906954
No.2: 情侣, Probability is 0.0648417
No.3: 好友, Probability is 0.0189635

实体1: 李晓华
实体2: 王大牛
李晓华和她的高中同学王大牛两个人前日一起去英国旅行。
关系是:
No.1: 好友, Probability is 0.526823
No.2: 兄弟姐妹, Probability is 0.177491
No.3: 夫妻, Probability is 0.132977
```

