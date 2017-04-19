#!/bin/bash

target=csv/jennis/training.csv
rm -fr $target
touch $target

head -n   10000 csv/jennis/361023.csv >> $target.tmp
head -n 1000000 csv/jennis/361024.csv >> $target.tmp

shuf $target.tmp > $target
#cat $target.tmp > $target

rm $target.tmp

echo "INFO: created training file $target"
