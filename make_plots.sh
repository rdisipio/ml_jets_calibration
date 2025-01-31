#!/bin/bash

INPUTFILE=$1
#user.sketabch.akt10.model.merged.histograms.root
#user.tnitta.akt10.dnn.largeR_substructure.histograms.root

./plot_response_largeR.py E  ${INPUTFILE} &
./plot_response_largeR.py pT ${INPUTFILE} &
./plot_response_largeR.py M  ${INPUTFILE} &
./plot_response_largeR.py eta ${INPUTFILE} &
wait

./plot_resolution.py E  ${INPUTFILE} &
./plot_resolution.py pT ${INPUTFILE} &
./plot_resolution.py eta  ${INPUTFILE} &
./plot_resolution.py M  ${INPUTFILE} &

wait



seq 0 5 | parallel -j 8 ./plot_response.py pT ptbin_{} ${INPUTFILE}
seq 0 5 | parallel -j 8 ./plot_response.py E  ptbin_{} ${INPUTFILE}
seq 0 5 | parallel -j 8 ./plot_response.py M  ptbin_{} ${INPUTFILE}

seq 0 3 | parallel -j 8 ./plot_response.py pT etabin_{} ${INPUTFILE}
seq 0 3 | parallel -j 8 ./plot_response.py E  etabin_{} ${INPUTFILE}
seq 0 3 | parallel -j 8 ./plot_response.py M  etabin_{} ${INPUTFILE}

seq 0 4 | parallel -j 8 ./plot_response.py pT massbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py E  massbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py M  massbin_{} ${INPUTFILE}


#for i in $(seq 0 4)
#do
#  for obs in pT E M
#  do
#     ./plot_response.py $obs ptbin_$i ${INPUTFILE}  &
#  done
#  wait
#done

#for i in $(seq 0 11)
#do
#  for obs in pT E M
#  do
#   ./plot_response.py $obs etabin_$i ${INPUTFILE} &
#  done
#  wait
#done

#for i in $(seq 0 4)
#do
#  for obs in pT E M
#  do
#   ./plot_response.py $obs massbin_$i ${INPUTFILE} &
#  done
#  wait
#done

