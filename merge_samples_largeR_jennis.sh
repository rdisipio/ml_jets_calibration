#!/bin/bash

target=csv/jennis/dijet
rm -fr $target.*.csv

#cat              csv/jennis/361021.csv >> $target.tmp
#cat              csv/jennis/361022.csv >> $target.tmp
#head -n      50000 csv/jennis/361023.csv >> $target.tmp

#57k
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp
#cat csv/jennis/361023.csv >> $target.tmp

head -n      50000 csv/jennis/361023.csv >> $target.tmp
head -n     100000 csv/jennis/361024.csv >> $target.tmp
head -n     100000 csv/jennis/361025.csv >> $target.tmp
head -n     100000 csv/jennis/361026.csv >> $target.tmp
head -n     100000 csv/jennis/361027.csv >> $target.tmp

shuf $target.tmp > $target.csv
rm $target.tmp

split --lines $(( $(wc -l < $target.csv) / 2)) $target.csv $target.

mv $target.aa $target.training.csv
mv $target.ab $target.testing.csv

head -n 100000 $target.testing.csv > $target.testing.quick.csv

echo "INFO: created training files: $target"
ls $target.*.csv
