#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import cgi
import BaseHTTPServer
import ssl

import tensorflow as tf
import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  return example

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  if len(sys.argv) > 2:
    port = int(sys.argv[2])
  else:
    port = None

  if len(sys.argv) > 3:
    # For https. See https://certbot.eff.org
    keyfile = sys.argv[3]
    certfile = sys.argv[4]
  else:
    keyfile = None
    certfile = None

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)

    if port is not None:
      class CorefRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
        def do_GET(self):
          idx = self.path.find("?")
          if idx >= 0:
            args = cgi.parse_qs(self.path[idx+1:])
            if "text" in args:
              text_arg = args["text"]
              if len(text_arg) == 1 and len(text_arg[0]) <= 10000:
                text = text_arg[0].decode("utf-8")
                print(u"Document text: {}".format(text))
                example = make_predictions(text, model)
                print_predictions(example)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write("jsonCallback({})".format(json.dumps(example)))
                return
          self.send_response(400)
          self.send_header("Content-Type", "application/json")
          self.end_headers()

      server = BaseHTTPServer.HTTPServer(("", port), CorefRequestHandler)
      if keyfile is not None:
        server.socket = ssl.wrap_socket(server.socket, keyfile=keyfile, certfile=certfile, server_side=True)
      print("Running server at port {}".format(port))
      server.serve_forever()
    else:
      while True:
        text = raw_input("Document text: ")
        print_predictions(make_predictions(text, model))
