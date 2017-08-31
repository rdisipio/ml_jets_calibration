#!/usr/bin/env python

import os, sys

from keras.models import load_model

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

from ROOT import TLorentzVector

np.set_printoptions( precision=4, suppress=True )

#################

from models import *
from features import *

#########

training_filename = sys.argv[1]

model_name = "4p_resnet"
if len(sys.argv) > 2:
   model_name = sys.argv[2]
print "INFO: Using model", model_name

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
n_input_all = len( features_all )

X_train_pT  = df_training[features_pT].values
X_train_eta = df_training[features_eta].values
X_train_E   = df_training[features_E].values
X_train_M   = df_training[features_M].values

print "INFO: features pT (%i):" % n_input_pT
print features_pT
print "INFO: features eta (%i):" % n_input_eta
print features_eta
print "INFO: features E (%i):" % n_input_E
print features_E
print "INFO: features M (%i):" % n_input_M
print features_M

do_preprocessing = "none"
if os.environ.has_key('PREPROCESSING'):
   do_preprocessing = os.environ['PREPROCESSING']

if do_preprocessing == "autoencoder":
   print "INFO: Preprocessing: standardization + autoencoder"
   # Set up scalers
   filename_scaler = "X_scaler.pkl"
   with open( filename_scaler, "rb" ) as file_scaler:
      X_scaler = pickle.load( file_scaler )
   print "INFO: X_scaler loaded from file", filename_scaler

   # Create autoencoder
   encoder = load_model( "encoder.h5" )
   encoding_dim = encoder.layers[-1].output_shape[1]
   print "INFO: loaded encoder %i -> %i" % ( n_input_all, encoding_dim )

   # these are the compressed data
   X_train_all = X_scaler.transform( X_train_all )
   X_train_all_encoded = encoder.predict(X_train_all)
   print "INFO: example of compressed data:"
   print X_train_all_encoded

elif do_preprocessing == "pca":
   # apply PCA
   print "INFO: Preprocessing: standardization + PCA"
   X_scaler = StandardScaler()
   X_train_all = X_scaler.fit_transform( X_train_all )
  
   print "INFO: Applying PCA"
   from sklearn.decomposition import PCA
   encoding_dim = 15
   pca = PCA(n_components=encoding_dim)
   X_train_all_encoded = pca.fit_transform(X_train_all)
   print "INFO: example of compressed data:"
   print X_train_all_encoded
else:
   print "INFO: Preprocessing: standardization"

   X_scaler_pT  = StandardScaler() #StandardScaler()
   X_train_pT   = X_scaler_pT.fit_transform( X_train_pT )

   X_scaler_eta = StandardScaler()
   X_train_eta  = X_scaler_eta.fit_transform( X_train_eta )

   X_scaler_E   = StandardScaler()
   X_train_E    = X_scaler_E.fit_transform( X_train_E )

   X_scaler_M   = StandardScaler()
   X_train_M    = X_scaler_M.fit_transform( X_train_M )

   X_train_all_encoded = [ X_train_pT, X_train_eta, X_train_E, X_train_M ]

#   filename_scaler = "X_scaler.pkl"
#   with open( filename_scaler, "rb" ) as file_scaler:
#      X_scaler = pickle.load( file_scaler )
#   print "INFO: X_scaler loaded from file", filename_scaler
#   X_scaler = StandardScaler()
#   X_train_all_encoded = X_scaler.fit_transform( X_train_all )
#   X_train_all = X_scaler.transform( X_train_all )
#   X_train_all_encoded = X_train_all
#   encoding_dim = len(X_train_all_encoded[0])

# now calibrate jets
y_train_all = df_training[ [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] ].values
y_scaler = StandardScaler()
#y_scaler = MinMaxScaler( (1,100) )
y_train_all = y_scaler.fit_transform( y_train_all )

y_weight_mc = df_training[ "mc_Weight" ].values

#~~~~~~~~~

BATCH_SIZE = 500
MAX_EPOCHS = 50
model_filename = "model.%s.h5" % model_name

callbacks_list = [ 
   # val_loss
#   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, epsilon=0.1, min_lr=0.001, verbose=1), 
#   EarlyStopping( monitor='val_loss', patience=5, mode='min', min_delta=0.0005, verbose=1 ),
#   ModelCheckpoint( model_filename + "-{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', mode='min', save_best_only=True, verbose=0), 
   ModelCheckpoint( model_filename, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
]

print "INFO: creating calibration DNN"
print "INFO: batch size:", BATCH_SIZE
print "INFO: max epochs:", MAX_EPOCHS

model_func = create_model_calib_4p_resnet
if model_name == "4p_resnet":
  model_func = create_model_calib_4p_resnet # best one so far
elif model_name == "4p":
  model_func = create_model_calib_4p
elif model_name == "SingleNet":
  model_func = create_model_calib_SingleNet
elif model_name == "433":
  model_func = create_model_calib_433
else:
  print "WARNING: unknown model function, using default 4p_resnet"

dnn = KerasRegressor( build_fn=model_func, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 )
#dnn = KerasRegressor( build_fn=create_model_calib_4, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 )#, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4x1, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 ) #, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4p_resnet, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 )#, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4x1_merged, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 ) #, sample_weight=y_weight_mc )

dnn.fit( [X_train_pT,X_train_eta,X_train_E,X_train_M], y_train_all )

#dnn.model.save( model_filename )
print "INFO: loss:"
print np.array( dnn.model.history.history['loss'] )
print

print "INFO: validation loss:"
print np.array( dnn.model.history.history['val_loss'] )
print

scaler_filename = "scaler.%s.pkl" % model_name
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump( X_scaler_pT,  file_scaler )
  pickle.dump( X_scaler_eta, file_scaler )
  pickle.dump( X_scaler_E,   file_scaler )
  pickle.dump( X_scaler_M,   file_scaler )
  pickle.dump( y_scaler,     file_scaler )

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

#with open( "history.pkl", "wb") as history_file:
#   pickle.dump( dnn, history_file )

# Print out example
for i in range(30):
  print "  ", y_nocalib[i], "----> DNN =", y_dnncalib[i], ":: Truth =", y_truth[i]

