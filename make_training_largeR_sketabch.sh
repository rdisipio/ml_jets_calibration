#!/bin/bash

target=csv/user.sketabch.akt10.csv
rm -fr $target
touch $target

head -n 200000 csv/sketabch/Pythia8EvtGen_JZ3W.txt >> $target
head -n 500000 csv/sketabch/Pythia8EvtGen_JZ4W.txt >> $target
head -n  50000 csv/sketabch/Pythia8EvtGen_JZ5W.txt >> $target

echo "INFO: created training file $target"
