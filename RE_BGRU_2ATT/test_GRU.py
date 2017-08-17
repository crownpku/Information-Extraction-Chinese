from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score
from checkpoint_compat import transform_name_var_dict

FLAGS = tf.app.flags.FLAGS


def main(_):
    # ATTENTION: change pathname before you load your model
    pathname = "./model/ATT_GRU_model-"

    wordembedding = np.load('./data/vec.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = 16693
    test_settings.num_classes = 12
    test_settings.big_num = 5561

    big_num_test = test_settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

           
            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)

            # ATTENTION: change the list to the iters you want to test !!
            #testlist = range(1000, 1800, 100)
            testlist = [1700]
            
            for model_iter in testlist:
                # for compatibility purposes only, name key changes from tf 0.x to 1.x, compat_layer
                saver.restore(sess, pathname + str(model_iter))


                time_str = datetime.datetime.now().isoformat()
                print(time_str)
                print('Evaluating all test data and save data for PR curve')

                test_y = np.load('./data/testall_y.npy')
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                order = np.argsort(-allprob)

                print('saving all test result...')
                current_step = model_iter

                # ATTENTION: change the save path before you save your result !!
                np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
                allans = np.load('./data/allans.npy')

                # caculate the pr curve area
                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))


def main_for_single_test():
    pathname = "./model/ATT_GRU_model-1700"
    wordembedding = np.load('./data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = 16693
    test_settings.num_classes = 12
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            while True:
                line = input("请输入测试句子:")
                #result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                #print(result)
            
    
    
    


if __name__ == "__main__":
    tf.app.run()
