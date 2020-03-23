#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Call with prefix to delete as argument"
  exit 1
fi

rm /workspace/results/history/$1*.csv

rm /workspace/results/trained_models/$1*.h5
rm /workspace/results/trained_models/$1*.pk

rm /workspace/results/major/$1/$1*.png
rm /workspace/results/major/$1/$1*.txt
rm /workspace/results/major/$1/$1*.h5
rm /workspace/results/major/$1/$1*.pk