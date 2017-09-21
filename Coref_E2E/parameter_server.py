#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
import util

if __name__ == "__main__":
  util.set_gpus()
  cluster_config = util.get_config("experiments.conf")[os.environ["EXP"]]["cluster"]
  cluster = tf.train.ClusterSpec(cluster_config["addresses"])
  server = tf.train.Server(cluster, job_name="ps", task_index=0)
  server.join()
