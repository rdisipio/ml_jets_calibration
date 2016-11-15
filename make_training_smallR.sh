#!/bin/bash

target=csv/user.mlisovyi.ca4.csv
rm -fr $target
touch $target

cat csv/user.mlisovyi.9682265._000003.361021.JZ1W.DAOD_JETM6.csv >>  ${target}
cat csv/user.mlisovyi.9682265._000004.361021.JZ1W.DAOD_JETM6.csv >>  ${target}
cat csv/user.mlisovyi.9682265._000005.361021.JZ1W.DAOD_JETM6.csv >>  ${target}

cat csv/user.mlisovyi.9682263._000005.361022.JZ2W.DAOD_JETM6.csv >> ${target}
cat csv/user.mlisovyi.9682263._000006.361022.JZ2W.DAOD_JETM6.csv >> ${target}

#head -n  500000 csv/user.mlisovyi.9682261._000001.361023.JZ3W.DAOD_JETM6.csv >> ${target}
#head -n  500000 csv/user.mlisovyi.9682261._000002.361023.JZ3W.DAOD_JETM6.csv >> ${target}
head -n 1500000 csv/user.mlisovyi.9682261._000003.361023.JZ3W.DAOD_JETM6.csv >> ${target}

head -n 1500000 csv/user.mlisovyi.9682259._000007.361024.JZ4W.DAOD_JETM6.csv >> ${target}

head -n  500000 csv/user.mlisovyi.9682258._000004.361025.JZ5W.DAOD_JETM6.csv >> ${target}

echo "INFO: created training file $target"
