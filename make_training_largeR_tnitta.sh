#!/bin/bash

target=csv/user.tnitta.akt10.csv
rm -fr $target
touch $target

head -n   500000  csv/user.tnitta.361023.DAOD_JETM8_p2666.csv >> $target.tmp
head -n  1000000  csv/user.tnitta.361024.DAOD_JETM8_p2666.csv >> $target.tmp
head -n   200000  csv/user.tnitta.361025.DAOD_JETM8_p2666.csv >> $target.tmp
head -n    50000  csv/user.tnitta.361026.DAOD_JETM8_p2666.csv >> $target.tmp
#head -n    10000  csv/user.tnitta.361027.DAOD_JETM8_p2666.csv >> $target.tmp

shuf $target.tmp > $target

rm $target.tmp

echo "INFO: created training file $target"
