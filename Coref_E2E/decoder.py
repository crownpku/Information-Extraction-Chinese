#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  output_filename = sys.argv[2]

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  model.load_eval_data()

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)

    with open(output_filename, "w") as f:
      for example_num, (tensorized_example, example) in enumerate(model.eval_data):
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
        predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
        example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
        example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
        example["head_scores"] = head_scores.tolist()
        f.write(json.dumps(example))
        f.write("\n")
        if example_num % 100 == 0:
          print "Decoded {} examples.".format(example_num + 1)
