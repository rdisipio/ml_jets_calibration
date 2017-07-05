#!/bin/bash

target=csv/jennis/training.csv
rm -fr $target
touch $target

#cat              csv/jennis/361021.csv >> $target.tmp
#cat              csv/jennis/361022.csv >> $target.tmp
head -n    300000 csv/jennis/361023.csv >> $target.tmp
head -n    300000 csv/jennis/361024.csv >> $target.tmp
head -n    400000 csv/jennis/361025.csv >> $target.tmp
head -n    500000 csv/jennis/361026.csv >> $target.tmp

shuf $target.tmp > $target

rm $target.tmp

echo "INFO: created training file $target"
