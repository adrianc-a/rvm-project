#!/bin/bash

DIRECTORY=data

if [ ! -d "$DIRECTORY" ]; then
  mkdir "$DIRECTORY"
fi

curl https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat -o data/airFoil.dat
curl https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data -o data/slumpTest.dat

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/banana.data.tar.gz -o data/banana.tar.gz
mkdir -p data/banana
tar xvzf data/banana.tar.gz -C data/banana
rm data/banana.tar.gz