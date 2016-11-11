#!/bin/bash

target=csv/user.mlisovyi.ca4.csv

cat csv/user.mlisovyi.9682265._000003.361021.JZ1W.DAOD_JETM6.csv >  ${target}
cat csv/user.mlisovyi.9682263._000003.361022.JZ2W.DAOD_JETM6.csv >> ${target}

head -n 500000 csv/user.mlisovyi.9682261._000001.361023.JZ3W.DAOD_JETM6.csv >> ${target}
head -n 500000 csv/user.mlisovyi.9682259._000002.output.csv >> ${target}
head -n 200000 csv/user.mlisovyi.9682258._000004.361025.JZ5W.DAOD_JETM6.csv >> ${target}
