#!/bin/bash

# File to curl data sets from the web

#__author__ = "Adrian Chiemelewski-Anders, Clara Tump, Bas Straathof \
#              and Leo Zeitler"

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

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/titanic.data.tar.gz -o data/titanic.tar.gz
mkdir -p data/titanic
tar xvzf data/titanic.tar.gz -C data/titanic
rm data/titanic.tar.gz
mv data/titanic/home/brain/raetsch/expose/titanic/* data/titanic
rm -r data/titanic/home

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/waveform.data.tar.gz -o data/waveform.tar.gz
mkdir -p data/waveform
tar xvzf data/waveform.tar.gz -C data/waveform
rm data/waveform.tar.gz
mv data/waveform/home/brain/raetsch/expose/waveform/* data/waveform
rm -r data/waveform/home

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/german.data.tar.gz -o data/german.tar.gz
mkdir -p data/german
tar xvzf data/german.tar.gz -C data/german
rm data/german.tar.gz
mv data/german/home/brain/raetsch/expose/german/* data/german
rm -r data/german/home

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/image.data.tar.gz -o data/image.tar.gz
mkdir -p data/image
tar xvzf data/image.tar.gz -C data/image
rm data/image.tar.gz
mv data/image/home/brain/raetsch/expose/image/* data/image
rm -r data/image/home

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/splice.data.tar.gz -o data/splice.tar.gz
mkdir -p data/splice
tar xvzf data/splice.tar.gz -C data/splice
rm data/splice.tar.gz
mv data/splice/home/brain/raetsch/expose/splice/* data/splice
rm -r data/splice/home

curl https://public.bmi.inf.ethz.ch/user/raetsch/benchmarks/thyroid.data.tar.gz -o data/thyroid.tar.gz
mkdir -p data/thyroid
tar xvzf data/thyroid.tar.gz -C data/thyroid
rm data/thyroid.tar.gz
mv data/thyroid/home/brain/raetsch/expose/thyroid/* data/thyroid
rm -r data/thyroid/home
