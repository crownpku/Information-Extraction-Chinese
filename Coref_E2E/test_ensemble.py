#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import collections
import operator

import numpy as np
import tensorflow as tf
import coref_model as cm
import util
import conll
import metrics

if __name__ == "__main__":
  if "GPU" in os.environ:
    util.set_gpus(int(os.environ["GPU"]))
  else:
    util.set_gpus()

  names = sys.argv[1:]
  print "Ensembling models from {}.".format(names)

  configs = util.get_config("experiments.conf")

  main_config = configs[names[0]]
  model = cm.CorefModel(main_config)
  model.load_eval_data()

  saver = tf.train.Saver()

  with tf.Session() as session:
    all_mention_scores = collections.defaultdict(list)

    for name in names:
      config = configs[name]
      log_dir = os.path.join(config["log_root"], name)
      checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
      print "Computing mention scores for {}".format(checkpoint_path)
      saver.restore(session, checkpoint_path)

      for example_num, (tensorized_example, example) in enumerate(model.eval_data):
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        mention_scores = session.run(model.predictions[2], feed_dict=feed_dict)
        all_mention_scores[example["doc_key"]].append(mention_scores)

        if example_num % 10 == 0:
          print "Computed {}/{} examples.".format(example_num + 1, len(model.eval_data))

    mean_mention_scores = { doc_key : np.mean(s, 0) for doc_key, s in all_mention_scores.items() }

    all_antecedent_scores = collections.defaultdict(list)
    mention_start_dict = {}
    mention_end_dict = {}
    antecedents_dict = {}

    for name in names:
      config = configs[name]
      log_dir = os.path.join(config["log_root"], name)
      checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
      print "Computing antecedent scores for {}".format(checkpoint_path)
      saver.restore(session, checkpoint_path)

      for example_num, (tensorized_example, example) in enumerate(model.eval_data):
        doc_key = example["doc_key"]
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        feed_dict[model.predictions[2]] = mean_mention_scores[doc_key]
        mention_starts, mention_ends, antecedents, antecedent_scores = session.run(model.predictions[3:7], feed_dict=feed_dict)
        if doc_key in mention_start_dict:
          assert (mention_starts == mention_start_dict[doc_key]).all()
          assert (mention_ends == mention_end_dict[doc_key]).all()
          assert (antecedents == antecedents_dict[doc_key]).all()
        else:
          mention_start_dict[doc_key] = mention_starts
          mention_end_dict[doc_key] = mention_ends
          antecedents_dict[doc_key] = antecedents

        all_antecedent_scores[doc_key].append(antecedent_scores)

        if example_num % 10 == 0:
          print "Computed {}/{} examples.".format(example_num + 1, len(model.eval_data))

    mean_antecedent_scores = { doc_key : np.mean(s, 0) for doc_key, s in all_antecedent_scores.items() }

    merged_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    for example_num, (tensorized_example, example) in enumerate(model.eval_data):
      doc_key = example["doc_key"]
      mention_starts = mention_start_dict[doc_key]
      mention_ends = mention_end_dict[doc_key]
      antecedents = antecedents_dict[doc_key]
      antecedent_scores = mean_antecedent_scores[doc_key]
      predicted_antecedents = []
      for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if index < 0:
          predicted_antecedents.append(-1)
        else:
          predicted_antecedents.append(antecedents[i, index])
      merged_predictions[doc_key] = model.evaluate_coref(mention_starts, mention_ends, predicted_antecedents, example["clusters"], coref_evaluator)

  conll_results = conll.evaluate_conll(main_config["conll_eval_path"], merged_predictions, official_stdout=True)
  average_f = sum(results["f"] for results in conll_results.values()) / len(conll_results)
  average_r = sum(results["r"] for results in conll_results.values()) / len(conll_results)
  average_p = sum(results["p"] for results in conll_results.values()) / len(conll_results)
  print "Merged average F1 (conll): {:.2f}%".format(average_f)
  print "Merged average Recall (conll): {:.2f}%".format(average_r)
  print "Merged average Precision (conll): {:.2f}%".format(average_p)
