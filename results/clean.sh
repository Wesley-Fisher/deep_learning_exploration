#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Call with prefix to delete as argument"
  exit 1
fi

rm /workspace/results/training_hist/$1*.pk
rm /workspace/results/models/$1*.h5
rm /workspace/results/history/$1*.csv