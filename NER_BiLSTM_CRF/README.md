## Recurrent neural networks for Chinese named entity recognition in TensorFlow
This repository contains a simple demo for chainese named entity recognition.

## Contributer
- [Jingyuan Zhang](https://github.com/zjy-ucas)
- [Mingjie Chen](https://github.com/superthierry)
- some data processing codes from [glample/tagger](https://github.com/glample/tagger)


## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- [jieba=0.37](https://github.com/fxsjy/jieba)


## Model
The model is a birectional LSTM neural network with a CRF layer. Sequence of chinese characters are projected into sequence of dense vectors, and concated with extra features as the inputs of recurrent layer, here we employ one hot vectors representing word boundary features for illustration. The recurrent layer is a bidirectional LSTM layer, outputs of forward and backword vectors are concated and projected to score of each tag. A CRF layer is used to overcome label-bias problem.

Our model is similar to the state-of-the-art Chinese named entity recognition model proposed in Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition.

## Basic Usage

### Default parameters:
- batch size: 20
- gradient clip: 5
- embedding size: 100
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001

Word vectors are trained with gensim version of word2vec on Chinese WiKi corpus, provided by [Chuanhai Dong](https://github.com/sea2603).

### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
```

### Online evaluate:
```shell
$ python3 main.py
```

## Suggested readings:
1. [Natural Language Processing (Almost) from Scratch](http://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf).  
Propose a unified neural network architecture for sequence labeling tasks.
2. [Neural Architectures for Named Entity Recognition](http://arxiv.org/abs/1603.01360).  
[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/lstm-cnn-crf.pdf).  
Combine Character-based word representations and word representations to enhance sequence labeling systems.
3. [Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks](http://www.cs.cmu.edu/~./wcohen/postscript/iclr-2017-transfer.pdf).  
[Multi-task Multi-domain Representation Learning for Sequence Tagging](http://xueshu.baidu.com/s?wd=paperuri%3A%288d2ae013d4ea38b3aba07a5f5cf8c8d1%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fpdf%2F1608.02689v1.pdf&ie=utf-8&sc_us=16810667041741374202).  
Transfer learning for sequence tagging.
4. [Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings](http://www.aclweb.org/website/anthology/D/D15/D15-1064.pdf).  
Propose a joint training objective for the embeddings that makes use of both (NER) labeled and unlabeled raw text
5. [Improving Named Entity Recognition for Chinese Social Media with Word Segmentation Representation Learning](http://anthology.aclweb.org/P/P16/P16-2025.pdf).  
[An Empirical Study of Automatic Chinese Word Segentation for Spoken Language Understanding and Named Entity Recognition](http://www.aclweb.org/anthology/N/N16/N16-1028.pdf).  
Using word segmentation outputs as additional features for sequence labeling syatems.
6. [Semi-supervised Sequence Tagging with Bidirectional Language Models](http://xueshu.baidu.com/s?wd=paperuri%3A%28e7dcf1a507dabc77f1e26c28068ca937%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fpdf%2F1705.0108&ie=utf-8&sc_us=17831018953161676191).  
State-of-the-art model on Conll03 NER task, adding pre-trained context embeddings from bidirectional language models for sequence labeling task.
7. [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](http://tcci.ccf.org.cn/conference/2016/papers/119.pdf).  
State-of-the-art model on SIGHAN2006 NER task.
8. [Named Entity Recognition with Bidirectional LSTM-CNNs](http://xueshu.baidu.com/s?wd=paperuri%3A%28995499661ccaa95ca3688318f4bc594b%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1511.08308&ie=utf-8&sc_us=14130444594064699095).  
Method to apply lexicon features.

