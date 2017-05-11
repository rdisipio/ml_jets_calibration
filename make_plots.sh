#!/bin/bash

INPUTFILE=training.model.merged.histograms.root
#user.sketabch.akt10.model.merged.histograms.root
#user.tnitta.akt10.dnn.largeR_substructure.histograms.root
[ ! -z $1 ] && INPUTFILE=$1

./plot_response_largeR.py E  ${INPUTFILE} &
./plot_response_largeR.py pT ${INPUTFILE} &
./plot_response_largeR.py M  ${INPUTFILE} &
./plot_response_largeR.py eta ${INPUTFILE} &

./plot_resolution.py E  ${INPUTFILE} &
./plot_resolution.py pT ${INPUTFILE} &
./plot_resolution.py M  ${INPUTFILE} &

wait

seq 0 4 | parallel -j 8 ./plot_response.py pT ptbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py E  ptbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py M  ptbin_{} ${INPUTFILE}

seq 0 11 | parallel -j 8 ./plot_response.py pT etabin_{} ${INPUTFILE}
seq 0 11 | parallel -j 8 ./plot_response.py E  etabin_{} ${INPUTFILE}
seq 0 11 | parallel -j 8 ./plot_response.py M  etabin_{} ${INPUTFILE}

