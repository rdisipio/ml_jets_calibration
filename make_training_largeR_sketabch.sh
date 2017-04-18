#!/bin/bash

target=csv/user.sketabch.akt10.csv
rm -fr $target
touch $target

head -n   100000 csv/sketabch/Pythia8EvtGen_JZ3W.txt >> $target.tmp
head -n  1000000 csv/sketabch/Pythia8EvtGen_JZ4W.txt >> $target.tmp
head -n  1000000 csv/sketabch/Pythia8EvtGen_JZ5W.txt >> $target.tmp

#cat csv/sketabch/Pythia8EvtGen_JZ3W.txt >> $target.tmp
#cat csv/sketabch/Pythia8EvtGen_JZ4W.txt >> $target.tmp
#cat csv/sketabch/Pythia8EvtGen_JZ5W.txt >> $target.tmp

shuf $target.tmp > $target
#cat $target.tmp > $target

rm $target.tmp

echo "INFO: created training file $target"
