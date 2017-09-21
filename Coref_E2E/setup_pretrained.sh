#!/bin/bash

curl -O http://appositive.cs.washington.edu/resources/coref/char_vocab.english.txt

curl -O http://appositive.cs.washington.edu/resources/coref/final.tgz
mkdir -p logs
tar -xzvf final.tgz -C logs
rm final.tgz
