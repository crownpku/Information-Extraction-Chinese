
# Chinese Relation Extraction by biGRU with Character and Sentence Attentions

### [中文Blog](http://www.crownpku.com//2017/08/19/%E7%94%A8Bi-GRU%E5%92%8C%E5%AD%97%E5%90%91%E9%87%8F%E5%81%9A%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%9A%84%E4%B8%AD%E6%96%87%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96.html)

Bi-directional GRU with Word and Sentence Dual Attentions for End-to End Relation Extraction

Original Code in https://github.com/thunlp/TensorFlow-NRE, modified for Chinese.

Original paper [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://anthology.aclweb.org/P16-2034) and [Neural Relation Extraction with Selective Attention over Instances](http://aclweb.org/anthology/P16-1200)

![](http://www.crownpku.com/images/201708/1.jpg)

![](http://www.crownpku.com/images/201708/2.jpg)


## Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

* scikit-learn (>=0.18)


## Usage


### * Training:

1. Prepare data in origin_data/ , including relation types (relation2id.txt), training data (train.txt), testing data (test.txt) and Chinese word vectors (vec.txt).

```
Current sample data includes the following 12 relationships:
unknown, 父母, 夫妻, 师生, 兄弟姐妹, 合作, 情侣, 祖孙, 好友, 亲戚, 同门, 上下级
```

2. Organize data into npy files, which will be save at data/
```
#python3 initial.py
```

3. Training, models will be save at model/
```
#python3 train_GRU.py
```


### * Inference

**If you have trained a new model, please remember to change the pathname in main_for_evaluation() and main() in test_GRU.py with your own model name.**

```
#python3 test_GRU.py
```

Program will ask for data input in the format of "name1 name2 sentence".

We have pre-trained model in /model. To test the pre-trained model, simply initialize the data and run test_GRU.py:

```
#python3 initial.py
#python3 test_GRU.py
```


## Sample Results

We make up some sentences and test the performance. The model gives good results, sometimes wrong but reasonable.

More data is needed for better performance.

```
INFO:tensorflow:Restoring parameters from ./model/ATT_GRU_model-9000
reading word embedding data...
reading relation to id

实体1: 李晓华
实体2: 王大牛
李晓华和她的丈夫王大牛前日一起去英国旅行了。
关系是:
No.1: 夫妻, Probability is 0.996217
No.2: 父母, Probability is 0.00193673
No.3: 兄弟姐妹, Probability is 0.00128172

实体1: 李晓华
实体2: 王大牛
李晓华和她的高中同学王大牛两个人前日一起去英国旅行。
关系是:
No.1: 好友, Probability is 0.526823
No.2: 兄弟姐妹, Probability is 0.177491
No.3: 夫妻, Probability is 0.132977

实体1: 李晓华
实体2: 王大牛
王大牛命令李晓华在周末前完成这份代码。
关系是:
No.1: 上下级, Probability is 0.965674
No.2: 亲戚, Probability is 0.0185355
No.3: 父母, Probability is 0.00953698

实体1: 李晓华
实体2: 王大牛
王大牛非常疼爱他的孙女李晓华小朋友。
关系是:
No.1: 祖孙, Probability is 0.785542
No.2: 好友, Probability is 0.0829895
No.3: 同门, Probability is 0.0728216

实体1: 李晓华
实体2: 王大牛
谈起曾经一起求学的日子，王大牛非常怀念他的师妹李晓华。
关系是:
No.1: 师生, Probability is 0.735982
No.2: 同门, Probability is 0.159495
No.3: 兄弟姐妹, Probability is 0.0440367

实体1: 李晓华
实体2: 王大牛
王大牛对于他的学生李晓华做出的成果非常骄傲！
关系是:
No.1: 师生, Probability is 0.994964
No.2: 父母, Probability is 0.00460191
No.3: 夫妻, Probability is 0.000108601

实体1: 李晓华
实体2: 王大牛
王大牛和李晓华是从小一起长大的好哥们
关系是:
No.1: 兄弟姐妹, Probability is 0.852632
No.2: 亲戚, Probability is 0.0477967
No.3: 好友, Probability is 0.0433101

实体1: 李晓华
实体2: 王大牛
王大牛的表舅叫李晓华的二妈为大姐
关系是:
No.1: 亲戚, Probability is 0.766272
No.2: 父母, Probability is 0.162108
No.3: 兄弟姐妹, Probability is 0.0623203

实体1: 李晓华
实体2: 王大牛
这篇论文是王大牛负责编程，李晓华负责写作的。
关系是:
No.1: 合作, Probability is 0.907599
No.2: unknown, Probability is 0.082604
No.3: 上下级, Probability is 0.00730342

实体1: 李晓华
实体2: 王大牛
王大牛和李晓华为谁是论文的第一作者争得头破血流。
关系是:
No.1: 合作, Probability is 0.819008
No.2: 上下级, Probability is 0.116768
No.3: 师生, Probability is 0.0448312
```

