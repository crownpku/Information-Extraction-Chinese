#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  name = os.environ["EXP"]
  task_index = int(os.environ["TASK"])
  config = util.get_config("experiments.conf")[name]
  report_frequency = config["report_frequency"]

  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  util.print_config(config)

  cluster_config = config["cluster"]

  util.set_gpus(cluster_config["gpus"][task_index])

  cluster = tf.train.ClusterSpec(cluster_config["addresses"])
  server = tf.train.Server(cluster,
                           job_name="worker",
                           task_index=task_index)

  # Assigns ops to the local worker by default.
  with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
    model = cm.CorefModel(config)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

  log_dir = config["log_dir"]
  is_chief = (task_index == 0)

  # Create a "supervisor", which oversees the training process.
  sv = tf.train.Supervisor(is_chief=is_chief,
                           logdir=log_dir,
                           init_op=init_op,
                           saver=saver,
                           global_step=model.global_step,
                           save_model_secs=120)

  # The supervisor takes care of session initialization, restoring from
  # a checkpoint, and closing when done or an error occurs.
  writer = tf.summary.FileWriter(os.path.join(log_dir, "w{}".format(task_index)), flush_secs=20)

  with sv.managed_session(server.target) as session:
    model.start_enqueue_thread(session)
    if "pretrained_dir" in config:
      ckpt = tf.train.get_checkpoint_state(config["pretrained_dir"])
      assert ckpt.model_checkpoint_path
      print "Loading pretrained model from {}".format(ckpt.model_checkpoint_path)
      saver.restore(session, ckpt.model_checkpoint_path)
      session.run(model.reset_global_step)
    accumulated_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, tf_loss, steps_per_second)
        accumulated_loss = 0.0
        writer.add_summary(util.make_summary({
          "Train Loss": average_loss,
          "Steps per second": steps_per_second
        }))

  # Ask for all the services to stop.
  sv.stop()
