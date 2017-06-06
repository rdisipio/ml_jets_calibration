#!/usr/bin/env python

import os, sys

from keras.models import load_model

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


early_stopping = EarlyStopping( monitor='val_loss', patience=3, mode='min' )

#checkpoint = ModelCheckpoint( "checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
callbacks_list = [ early_stopping ]
#callbacks_list = []

#################

from models import *
from features import *

training_filename = sys.argv[1]

# Set up scalers
filename_scaler = "X_scaler.pkl"
with open( filename_scaler, "rb" ) as file_scaler:
   X_scaler = pickle.load( file_scaler )

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
X_train_all = X_scaler.fit_transform( X_train_all )

# Create autoencoder
n_input_all = len( features_all )
#encoding_dim = 10
encoder = load_model( "encoder.h5" )
encoding_dim = encoder.encoding_dim
print "INFO: loaded encoder %i -> %i" % ( n_input_all, encoding_dim )

# these are the compressed data
X_train_all_encoded = encoder.predict(X_train_all)
print "INFO: example of compressed data:"
print X_train_all_encoded

# now calibrate jets
y_train_all = df_training[ [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] ].values

y_scaler = StandardScaler()
y_train_all = y_scaler.fit_transform( y_train_all )

def create_model_calib4():
   input_calib = Input( shape=(encoding_dim, ))
   dnn_calib   = Dense( 300 )(input_calib)
   dnn_calib   = Dense( 200 )(dnn_calib)
   dnn_calib   = Dense( 100 )(dnn_calib)
   dnn_calib   = Dense(  50 )(dnn_calib)
   dnn_calib   = Dense(   4 )(dnn_calib)
   dnn_model   = Model( inputs=input_calib, outputs=dnn_calib )
   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
   print "INFO: DNN calibration model compiled"
   return dnn_model

BATCH_SIZE = 10000
MAX_EPOCHS = 10
print "INFO: creating calibration DNN"
dnn = KerasRegressor( build_fn=create_model_calib4, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05, callbacks=callbacks_list, verbose=1 )
dnn.fit( X_train_all_encoded, y_train_all )

model_filename = "model.calib4.h5" 
dnn.model.save( model_filename )

scaler_filename = "scaler.largeR_substructure.pkl"
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump( X_scaler, file_scaler )
  pickle.dump( y_scaler, file_scaler )

#  pickle.dump( poly,   file_scaler )
print "INFO: scalers saved to file", scaler_filename
print "INFO: model saved to file",   model_filename

print "INFO: testing..."
y_nocalib = df_training[y_features_nocalib].values
y_truth   = df_training[y_features_truth].values
y_calib   = df_training[y_features_calib].values

y_dnncalib = dnn.predict( X_train_all_encoded )
y_dnncalib = y_scaler.inverse_transform( y_dnncalib )
print

# Print out example
for i in range(30):
  print "  ", y_nocalib[i], "----> DNN =", y_dnncalib[i], ":: Truth =", y_truth[i]

