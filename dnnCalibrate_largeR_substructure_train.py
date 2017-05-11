#!/usr/bin/env python

import os, sys

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score

import sklearn.utils

try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
import pandas as pd

from ROOT import *

np.set_printoptions( precision=2, suppress=True )

early_stopping = EarlyStopping( monitor='val_loss', patience=20, mode='min' )

#checkpoint = ModelCheckpoint( "checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

callbacks_list = [ early_stopping ]

#################

from models import *
from features import *

training_filename = sys.argv[1]

# Set up scalers
create_scaler = StandardScaler
#create_scaler = MinMaxScaler
scaler_pT  = create_scaler()
scaler_eta = create_scaler()
scaler_E   = create_scaler()
scaler_M   = create_scaler() 
scaler_all = create_scaler()
#poly = PolynomialFeatures(2)

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
X_train_all = scaler_all.fit_transform( X_train_all )

X_train_pT = df_training[features_pT].values
X_train_pT = scaler_pT.fit_transform( X_train_pT )

X_train_eta = df_training[features_eta].values
X_train_eta = scaler_eta.fit_transform( X_train_eta )

X_train_E = df_training[features_E].values
X_train_E = scaler_E.fit_transform( X_train_E )

X_train_M = df_training[features_M].values
X_train_M = scaler_M.fit_transform( X_train_M )

y_train = df_training[ [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] ].values

MAX_EPOCHS = 30
BATCH_SIZE = 10000

dnn = KerasRegressor( build_fn=create_model_merged, nb_epoch=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05, callbacks=callbacks_list, verbose=1 )
dnn.fit( [ X_train_pT, X_train_eta, X_train_E, X_train_M ], y_train )

res = dnn.score( [ X_train_pT, X_train_eta, X_train_E, X_train_M ], y_train ) 
print "Score(pT,E):", res
dnn.model.save_weights( "weights.model_merged.h5", True )
model_filename = "model.merged.h5" 
dnn.model.save( model_filename )

#y_train1 = df_training[ [ "jet_truth_Pt", "jet_truth_E" ] ].values
#dnn1 = KerasRegressor( build_fn=create_model_pT_E_parallel, nb_epoch=20, batch_size=1000, verbose=1 )
#dnn1.fit( [ X_train_pT, X_train_E ], y_train1 )
#res1 = dnn.score( [ X_train_pT, X_train_E ], y_train1 ) 
#print "Score(pT,E):", res1

#dnn1 = KerasRegressor( build_fn=create_model_pT_E_single, nb_epoch=20, batch_size=1000, verbose=1 )
#dnn1.fit( X_train_pT, y_train1 )
#res1 = dnn1.score( [ X_train_pT ], y_train1 ) 
#print "Score(pT+E):", res1

#dnn1.model.save_weights( "weights.model_pT_E_parallel.h5", True )

#dnn.model.save( "dnn.largeR_substructure.sketabch.h5" )

scaler_filename = "scaler.largeR_substructure.pkl"
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump( scaler_pT,  file_scaler )
  pickle.dump( scaler_eta, file_scaler )
  pickle.dump( scaler_E,   file_scaler )
  pickle.dump( scaler_M,   file_scaler )
  pickle.dump( scaler_all, file_scaler )

#  pickle.dump( poly,   file_scaler )
print "INFO: scalers saved to file", scaler_filename
print "INFO: model saved to file", model_filename

print "INFO: testing..."
y_nocalib = df_training[y_features_nocalib].values
y_truth   = df_training[y_features_truth].values
y_calib   = df_training[y_features_calib].values
y_dnncalib = dnn.predict( [ X_train_pT, X_train_eta, X_train_E, X_train_M ] )
print

# Print out example
for i in range(30):
  print "  ", y_nocalib[i], "----> DNN =", y_dnncalib[i], ":: Truth =", y_truth[i]

