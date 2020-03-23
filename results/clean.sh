#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Call with prefix to delete as argument"
  exit 1
fi

rm /workspace/results/$1/$1*.csv
rm /workspace/results/$1/$1*.png
rm /workspace/results/$1/$1*.txt
rm /workspace/results/$1/$1*.h5
rm /workspace/results/$1/$1*.pk