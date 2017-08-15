# encoding:utf-8

import jieba.posseg as pseg
import os
import os.path
import cPickle
import jieba

pickleDir = "pickle"
dataDir = "data"


def load_stopwords():
    with open(os.path.join(dataDir, 'stopwords.txt'), 'r') as stopwords_file:
        stopwords = []
        for line in stopwords_file:
            stopwords.append(line.strip())

        return stopwords


# 构建字典:词到id、词性到id的映射
# save_num:按照词频，保留下save_num个词语（去停用词）
def generateDic2(sentence_filepath, save_num=15000):
    jieba.load_userdict(os.path.join(dataDir, 'people.txt'))
    jieba.load_userdict(os.path.join(dataDir, 'stopwords.txt'))

    with open(sentence_filepath, 'r') as sentence_file, \
            open(os.path.join(pickleDir, 'word2id_dic2.pkl'), 'wb') as word2id_dic_file, \
            open(os.path.join(pickleDir, 'pos2id_dic2.pkl'), 'wb') as pos2id_dic_file:

        stopwords = load_stopwords()

        wordfeq_dict = dict()

        word2id_dic = dict()
        pos2id_dic = dict()

        word2id = 0
        pos2id = 0

        line_idx = 0
        for sentence in sentence_file:
            sentence = sentence.strip()
            # print sentence
            words = pseg.cut(sentence)
            for w in words:
                word, pos = w.word, w.flag
                word = word.encode('utf-8')
                pos = pos.encode('utf-8')
                if word in stopwords:
                    continue
                else:

                    wordfeq_dict.setdefault(word, 0)
                    wordfeq_dict[word] += 1

                    if pos not in pos2id_dic:
                        pos2id_dic[pos] = pos2id
                        pos2id += 1
            line_idx += 1
            if line_idx % 5000 == 0:
                print line_idx

        sorted_wordfeq = sorted(wordfeq_dict.items(), key=lambda item: item[1], reverse=True)

        word_id = 0
        for item in sorted_wordfeq[0:save_num]:
            word = item[0]
            word2id_dic[word] = word_id
            word_id += 1

        cPickle.dump(word2id_dic, word2id_dic_file)
        cPickle.dump(pos2id_dic, pos2id_dic_file)


# 对齐
# 生成训练集和测试集
def align(sentence_filepath, train_filepath, peopleset_filepath):
    jieba.load_userdict(os.path.join(dataDir, 'people.txt'))

    with open(sentence_filepath, 'r') as sentence_file, open(train_filepath, 'r') as train_r_file, \
            open(peopleset_filepath, 'r') as peopleset_file, \
            open(os.path.join(dataDir, 'train.txt'), 'w') as train_file, \
            open(os.path.join(dataDir, 'test.txt'), 'w') as test_file:

        train_r_dict = dict()

        # loading train relation
        # 训练集 关系对
        for line in train_r_file:
            line = line.strip()
            entry = line.split('\t')
            p1, p2, relation = entry[0], entry[1], entry[2]
            train_r_dict[p1 + ',' + p2] = relation

        peopleset = set()

        # 人名~
        for line in peopleset_file:
            line = line.strip()
            peopleset.add(line)

        for line in sentence_file:
            line = line.strip()

            seg = jieba.cut(line)
            peopleset_line = set()
            for word in seg:
                word = word.encode('utf-8')
                if word in peopleset:
                    peopleset_line.add(word)

            peoplelist_line = (list)(peopleset_line)

            for i in range(len(peoplelist_line)):
                p1 = peoplelist_line[i]
                for j in range(len(peoplelist_line)):
                    if i != j:
                        p2 = peoplelist_line[j]
                        if p1 + ',' + p2 in train_r_dict:
                            relation = train_r_dict[p1 + ',' + p2]
                            train_file.write(p1 + '\t' + p2 + '\t' + relation + '\t' + line + '\n')
                        else:
                            test_file.write(p1 + '\t' + p2 + '\tunknown\t' + line + '\n')


relation2id_dic = {'父母': 0, '夫妻': 1, '师生': 2, '兄弟姐妹': 3, \
                   '合作': 4, '情侣': 5, '祖孙': 6, '好友': 7, '亲戚': 8, '同门': 9, '上下级': 10, 'unknown': -1}

id2relation_dic = {'0': '父母', '1': '夫妻', '2': '师生', '3': '兄弟姐妹', \
                   '4': '合作', '5': '情侣', '6': '祖孙', '7': '好友', '8': '亲戚', '9': '同门', '10': '上下级', '-1': 'unknown'}


def load_word2iddic():
    with open(os.path.join(pickleDir, 'word2id_dic2.pkl'), 'rb') as word2id_picklefile:
        return cPickle.load(word2id_picklefile)


def load_pos2iddic():
    with open(os.path.join(pickleDir, 'pos2id_dic2.pkl'), 'rb') as pos2id_picklefile:
        return cPickle.load(pos2id_picklefile)


# 以实体对单位抽取特征，抽取实体1 2 前后win个词语的词、词性，使用one hot表示
def feature_extract2(filepath, win=3):
    jieba.load_userdict(os.path.join(dataDir, 'people.txt'))

    word2id_dic = load_word2iddic()
    unknown_word_id = len(word2id_dic)

    print('word2id_dic size %d' % unknown_word_id)

    pos2id_dic = load_pos2iddic()
    unknown_pos_id = len(pos2id_dic)
    print('pos2id_dic size %d' % unknown_pos_id)

    with open(filepath, 'r') as file, \
            open(os.path.join(dataDir, 'feature2_' + os.path.split(filepath)[-1]), 'w') as feature_file, \
            open(os.path.join(dataDir, 'entitypair2_' + os.path.split(filepath)[-1]), 'w') as entitypair_file:

        words_feature_dict = dict()
        pos_feature_dict = dict()
        relation_dict = dict()

        stopwords = load_stopwords()
        line_idx = 0

        for line in file:
            line = line.strip()

            p1, p2, relation, sentence = line.split('\t')

            relation_id = relation2id_dic[relation]

            words_feature_dict.setdefault(p1, dict())
            words_feature_dict[p1].setdefault(p2, [0] * (unknown_word_id + 1))

            pos_feature_dict.setdefault(p1, dict())
            pos_feature_dict[p1].setdefault(p2, [0] * (unknown_pos_id + 1))
            relation_dict.setdefault(p1, dict())
            relation_dict[p1][p2] = relation_id

            words = pseg.cut(sentence)

            word_list = []
            pos_list = []

            i = 0
            e1_idx = -1
            e2_idx = -1

            lenOfSentence = 0

            for w in words:
                word, pos = w.word, w.flag

                word = word.encode('utf-8')
                pos = pos.encode('utf-8')
                word_list.append(word)
                pos_list.append(pos)

                if word == p1:
                    e1_idx = i
                elif word == p2:
                    e2_idx = i

                i += 1

            lenOfSentence = i

            if e1_idx == -1 or e2_idx == -1:

                # print line
                pass

            else:

                for word_idx in range(e1_idx - win, e1_idx + win + 1):
                    if word_idx == e1_idx:
                        continue
                    # 超过范围
                    if word_idx < 0 or word_idx >= lenOfSentence:

                        pass

                    elif word_list[word_idx] in stopwords:
                        pass

                    else:

                        word_uni = word_list[word_idx]
                        pos_uni = pos_list[word_idx]

                        if word_uni not in word2id_dic:

                            # words_feature.append((str)(unknown_word_id))
                            words_feature_dict[p1][p2][unknown_word_id] = 1

                        else:
                            # words_feature.append((str)(word2id_dic[word_uni]))

                            words_feature_dict[p1][p2][word2id_dic[word_uni]] = 1

                        if pos_uni not in pos2id_dic:
                            pos_feature_dict[p1][p2][unknown_pos_id] = 1
                        else:
                            pos_feature_dict[p1][p2][pos2id_dic[pos_uni]] = 1

            line_idx += 1
            if line_idx % 5000 == 0:
                print '已经处理%d行' % line_idx

        for p1, tmp_dict in words_feature_dict.items():
            for p2, words_feature_list in words_feature_dict[p1].items():

                relation_id = relation_dict[p1][p2]
                pos_feature_list = pos_feature_dict[p1][p2]

                entitypair_file.write(p1 + ' ' + p2 + '\n')
                feature_file.write((str)(relation_id))
                feature_id = 1
                for feature in words_feature_list:
                    if feature != 0:
                        feature_file.write(' ' + str(feature_id) + ':' + str(feature))
                    feature_id += 1

                for feature in pos_feature_list:
                    if feature != 0:
                        feature_file.write(' ' + str(feature_id) + ':' + str(feature))
                    feature_id += 1

                feature_file.write('\n')


# 根据libsvm的预测结果整理，得到预测结果
def handle_libsvm_result(predict_filepath, entitypair_filepath):
    with open(predict_filepath, 'r') as predict_file, open(entitypair_filepath, 'r') as entitypair_file, \
            open(os.path.join(dataDir, 'rsl_' + os.path.split(predict_filepath)[-1]), 'w') as rsl_file:
        line_idx = 0

        line = predict_file.readline()
        line = predict_file.readline()

        entitypair_line = entitypair_file.readline()

        while line != '':
            line = line.strip()

            entry = line.split(' ')
            predict_label = entry[0]

            relation = id2relation_dic[predict_label]
            p1, p2 = entitypair_line.strip().split(' ')
            rsl_file.write(p1 + '\t' + p2 + '\t' + relation + '\n')

            entitypair_line = entitypair_file.readline()
            line = predict_file.readline()


def read_relation(relation_filepath):
    with open(relation_filepath, 'r') as relation_file:
        relation_dict = dict()
        for line in relation_file:
            p1, p2, relation = line.strip().split('\t')
            relation_dict[p1 + ',' + p2] = relation

        return relation_dict


# 评测 只对在test_relation.txt里的关系判断对错。
# 准确率 = 在test_relation里的且预测准确的/在test_relation里的关系对总数
# 召回率 = 在test_relation里的且预测准确的/test_relation的关系对总数
def evaluation(rsl_filepath, reference_filepath):
    # pass

    rsl_relation_dict = read_relation(rsl_filepath)
    reference_relation_dict = read_relation(reference_filepath)

    right_c = 0
    return_c = 0

    for entitypair, relation in rsl_relation_dict.items():
        if entitypair in reference_relation_dict:
            return_c += 1
            if reference_relation_dict[entitypair] == relation:
                right_c += 1

    # print return_c
    precious = right_c * 1.0 / return_c
    recall = right_c * 1.0 / len(reference_relation_dict)

    return precious, recall


if __name__ == "__main__":
    print ('--align,and generate trainset and testset.')
    align('data/sentence.txt', 'data/train_relation.txt', 'data/people.txt')

    print ('--generate dic.')
    generateDic2('data/sentence.txt')

    print ('--feature extract.')
    feature_extract2('data/train.txt')
    feature_extract2('data/test.txt')

    # 在使用libsvm进行训练和预测以后进行结果整理和评测
    # print '--generate result and evaluate.'
    # handle_libsvm_result('data/predict.txt','data/entitypair2_test.txt')
    # print evaluation('data/rsl_predict.txt','data/test_relation.txt')
