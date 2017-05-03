Setup Environment
=================

Follow the instructions on the ATLAS ML twiki page
https://twiki.cern.ch/twiki/bin/view/AtlasComputing/MLSoftwareStandAloneSetup

```
module load anaconda2
source activate machine_learning
```

Go to your work directory
```
cd MLjetsCalib
```

Quick Run
=========
```bash

mkdir csv
mkdir csv/jennis
scp lxplus.cern.ch:/afs/cern.ch/work/s/sketabch/public/ML_input/*.txt ./csv/jennis

./make_training_largeR_jennis.sh # this creates csv/jennis/training.csv with shuffled entries

./dnnCalibrate_largeR_serial_train.py csv/jennis/training.csv
./dnnCalibrate_largeR_substructure_predict.py csv/jennis/training.csv
./make_plots.sh training.model.merged.histograms.root
```

You may want to modify ```features.py``` and ```models.py``` .

Setup EOS
=========

```bash
mkdir -p local_eos
export EOS_MGM_URL=root://eosatlas.cern.ch
kinit ${USER}@CERN.CH
eos fuse mount local_eos
```

You can now access files, e.g.:
```
ls local_eos/atlas/atlascerngroupdisk/perf-jets/JSS/MVACalibration/
```

When you're done:
```
eos fuse umount local_eos
```

Make input CSV files
====================

```
mkdir csv
```

Assuming ntuples are under ```${PWD}/ntuples/``` e.g. via symlink

```
./makeInput_smallR.py ntuples/user.mlisovyi.361024.JZ4W.DAOD_JETM6.e3668_s2576_s2132_r7725_r7676_p2719_EXT0.smallR_20161020_v1_output.root/user.mlisovyi.9682259._000001.output.root

# content of file called tatsumi_qcd.dat
user.tnitta.361023.DAOD_JETM8_p2666.dat
user.tnitta.361024.DAOD_JETM8_p2666.dat
user.tnitta.361025.DAOD_JETM8_p2666.dat
user.tnitta.361026.DAOD_JETM8_p2666.dat
user.tnitta.361027.DAOD_JETM8_p2666.dat
```

Each of these file contain the paths of .root files (ntuples)
```
/home/r/rorr/disipio/development/MLJetsCalib/ntuples/tatsumi_qcd/JETM8/user.tnitta.361027.DAOD_JETM8_p2666.v19heavy_ntuple.root/user.tnitta.9682737._000001.ntuple.root
```

```
cat tatsumi_qcd.dat | parallel -j 8 ./makeInput_largeR.py {}
./make_training_largeR.sh
```

# etc..for all files (maybe use GNU parallel)

* Train network (pT,eta,E) -> (pT,E)
Assume (eta,phi) measured perfectly

```
./dnnCalibrate_smallR_train.py csv/user.mlisovyi.9682259._000005.output.csv 
./dnnCalibrate_largeR_substructure_train.py csv/user.tnitta.akt10.csv
```

This will create a file called dnn.pT_E.h5 (containig the DNN) and a file called scaler.pkl (containing the StandardScaler used to preprocess the input)

Predict output (testing) 
========================

```
./dnnCalibrate_smallR_predict.py csv/user.mlisovyi.9682259._000005.output.csv dnn.pT_E.h5
./dnnCalibrate_largeR_substructure_predict.py csv/user.tnitta.akt10.csv
```

This will create a ROOT file called user.mlisovyi.9682259._000005.dnn.pT_E.histograms.root
Repeat this for all csv files (maybe use GNU parallel)

# On full datasets
```
cat user.tnitta.csv.dat | parallel -j 8 ./dnnCalibrate_largeR_predict.py {}

hadd -f user.tnitta.all.DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root user.tnitta.36102*DAOD_JETM8_p2666.dnn.largeR.pT_M.histograms.root
```

Make plots
==========

The output file of the previous step contains response histograms for several eta regions indexed as 0...N

```
mkdir img

./plot_response.py pT 0
./plot_response.py pT 1
...

./plot_response.py E 0
./plot_response.py E 1
...
```

###############
# large-R jets

```
INPUTFILE=user.tnitta.akt10.dnn.largeR_substructure.histograms.root

./plot_response_largeR.py E  ${INPUTFILE}
./plot_response_largeR.py pT ${INPUTFILE}
./plot_response_largeR.py M  ${INPUTFILE}

seq 0 4 | parallel -j 8 ./plot_response.py pT ptbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py E  ptbin_{} ${INPUTFILE}
seq 0 4 | parallel -j 8 ./plot_response.py M  ptbin_{} ${INPUTFILE}

seq 0 11 | parallel -j 8 ./plot_response.py pT etabin_{} ${INPUTFILE}
seq 0 11 | parallel -j 8 ./plot_response.py E  etabin_{} ${INPUTFILE}
seq 0 11 | parallel -j 8 ./plot_response.py M  etabin_{} ${INPUTFILE}
```

Do you need GPUs?
==================

```
ssh gravity01
source setup_machine_learning.sh 
module load gcc/4.8.1
module load cuda/7.5

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python ./dnnCalibrate_largeR_substructure_train.py csv/jennis/training.csv
```

To submit to SciNet/Gravity GPU cluster
```
qsub -l nodes=1:ppn=12:gpus=2,walltime=12:00:00 -q gravity -I
```
