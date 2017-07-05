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

np.set_printoptions( precision=2, suppress=True )

#################

from models import *
from features import *

#########

training_filename = sys.argv[1]

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
n_input_all = len( features_all )

if os.environ.has_key('USE_AUTOENCODER'):
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

if os.environ.has_key('USE_PCA'):
   # apply PCA
   X_scaler = StandardScaler()
   X_train_all = X_scaler.fit_transform( X_train_all )
  
   print "INFO: Applying PCA"
   from sklearn.decomposition import PCA
   encoding_dim = 15
   pca = PCA(n_components=encoding_dim)
   X_train_all_encoded = pca.fit_transform(X_train_all)

print "INFO: example of compressed data:"
print X_train_all_encoded

# now calibrate jets
y_train_all = df_training[ [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] ].values
y_weight_mc = df_training[ "mc_Weight" ].values

y_scaler = StandardScaler()
y_train_all = y_scaler.fit_transform( y_train_all )

# Apply oversampling/SMOTE
if os.environ.has_key('APPLY_SMOTE'):
   try:
      from imblearn.over_sampling import SMOTE, RandomOverSampler
   except:
      print "ERROR: imblearn library not installed. See page http://contrib.scikit-learn.org/imbalanced-learn/"
   sm = SMOTE(kind='regular')
   X_train_all_encoded, y_train_all = sm.fit_sample( X_train_all_encoded, y_train_all )

#~~~~~~~~~

def create_model_calib_4():
   input_calib = Input( shape=(encoding_dim, ))

   dnn_calib   = Dense( 500, activation='relu' )(input_calib)
   dnn_calib   = Dense( 300, activation='relu' )(dnn_calib)
   dnn_calib   = Dense( 200, activation='relu' )(dnn_calib)
   dnn_calib   = Dense( 100, activation='relu' )(dnn_calib)
   dnn_calib   = Dense(  50, activation='relu' )(dnn_calib)
   dnn_calib   = Dense(   4 )(dnn_calib)
   dnn_model   = Model( inputs=input_calib, outputs=dnn_calib )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
   print "INFO: DNN calibration model compiled"
   return dnn_model

#~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_model_calib_4x1():
   input_calib = Input( shape=(encoding_dim, ))

#   x_pT_eta_E_M = Dense(500, activation='relu')(input_calib)
#   x_pT_eta_E_M = Dense(300, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense(100, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 50, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense(  4)(x_pT_eta_E_M)
   
   dnn_calib_pT = Dense( 500 )(input_calib)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense( 300)(dnn_calib_pT) 
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense( 100)(dnn_calib_pT)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense(  50)(dnn_calib_pT)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense(1)(dnn_calib_pT)

   dnn_calib_eta = Dense( 500 )(input_calib)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense( 300)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense( 100)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense(  50)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense(1)(dnn_calib_eta)

   dnn_calib_E = Dense( 500 )(input_calib)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense( 300)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense( 100)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense(  50)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense(1)(dnn_calib_E)

   dnn_calib_M = Dense( 500 )(input_calib)
   dnn_calib_M = Activation('relu')(dnn_calib_M)
   dnn_calib_M = Dense( 300)(dnn_calib_M)
   dnn_calib_M = Activation('relu')(dnn_calib_M)
   dnn_calib_M = Dense( 100)(dnn_calib_M)
   dnn_calib_M = Activation('relu')(dnn_calib_M)
   dnn_calib_M = Dense(  50)(dnn_calib_M)
   dnn_calib_M = Activation('relu')(dnn_calib_M)
   dnn_calib_M = Dense(1)(dnn_calib_M)

#   dnn_calib_pT_E   = concatenate( [ dnn_calib_pT, dnn_calib_E ] )
#   dnn_calib_pT_M   = concatenate( [ dnn_calib_pT, dnn_calib_M ] )
#   dnn_calib_eta_E  = concatenate( [ dnn_calib_eta, dnn_calib_E ] )
#   dnn_calib_eta_M  = concatenate( [ dnn_calib_eta, dnn_calib_M ] )

#   dnn_calib_pT_eta_E = concatenate( [ dnn_calib_pT_E, dnn_calib_eta_E ] )
#   dnn_calib_pT_eta_E = Dense(3)(dnn_calib_pT_eta_E)
    
#   dnn_calib_pT_eta_M = concatenate( [ dnn_calib_pT_M, dnn_calib_eta_M ] )
#   dnn_calib_pT_eta_M = Dense(3)(dnn_calib_pT_eta_M)
   
   dnn_calib_pT_eta_E = concatenate( [ dnn_calib_pT, dnn_calib_eta, dnn_calib_E ] )
#   dnn_calib_pT_eta_E = Dense(6, activation='linear')(dnn_calib_pT_eta_E)
#   dnn_calib_pT_eta_E = Dense(6, activation='linear')(dnn_calib_pT_eta_E)
#   dnn_calib_pT_eta_E = Dense(6, activation='linear')(dnn_calib_pT_eta_E)
 
   dnn_calib_pT_eta_M = concatenate( [ dnn_calib_pT, dnn_calib_eta, dnn_calib_M ] )
#   dnn_calib_pT_eta_M = Dense(6, activation='linear')(dnn_calib_pT_eta_M)
#   dnn_calib_pT_eta_M = Dense(6, activation='linear')(dnn_calib_pT_eta_M)
#   dnn_calib_pT_eta_M = Dense(6, activation='linear')(dnn_calib_pT_eta_M)

   dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )
   dnn_calib_pT_eta_E_M = Dense(4, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(200, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(100, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(4)(dnn_calib_pT_eta_E_M)

 #  dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT, dnn_calib_eta, dnn_calib_E, dnn_calib_M ] )
#   dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT_eta_E, dnn_calib_M ] )

#   dnn_calib_pT_eta_E_M = average( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )
#   dnn_calib_pT_eta_E_M = maximum( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )
   
#   calibrated = average( [ dnn_calib_pT_eta_E_M, x_pT_eta_E_M ] )

   calibrated = Dense( 4, activation='linear', name='calibrated' )(dnn_calib_pT_eta_E_M)

   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
   print "INFO: DNN calibration model compiled"
   return dnn_model

def create_model_calib_resnet():
   def _block(_input ):
      n = K.int_shape(_input)[1]
      _output = Dense(n)(_input)
      #_output = BatchNormalization()(_output)
      _output = Activation('relu')(_output)
      _output = Dense(n)(_output)
      _output = add( [ _output, _input ] )
      _output = Activation('relu')(_output)
      return _output
   
   input_calib = Input( shape=(encoding_dim, ))

   tower_pT = Dense(100, kernel_initializer='glorot_normal' )(input_calib)
   tower_pT = _block(tower_pT)
   tower_pT = Dense(50)(tower_pT)
   tower_pT = _block(tower_pT)
   tower_pT = Dense(10)(tower_pT)
   tower_pT = _block(tower_pT)
   tower_pT = Dense(1)(tower_pT)
   
   tower_eta = Dense(100, kernel_initializer='glorot_normal' )(input_calib)
   tower_eta = _block(tower_eta)
   tower_eta = Dense(50)(tower_eta)
   tower_eta = _block(tower_eta)
   tower_eta = Dense(10)(tower_eta)
   tower_eta = _block(tower_eta)
   tower_eta = Dense(1)(tower_eta)

   tower_E = Dense(100, kernel_initializer='glorot_normal' )(input_calib)
   tower_E = _block(tower_E)
   tower_E = Dense(50)(tower_E)
   tower_E = _block(tower_E)
   tower_E = Dense(10)(tower_E)
   tower_E = _block(tower_E)
   tower_E = Dense(1)(tower_E)

   tower_M = Dense(100, kernel_initializer='glorot_normal' )(input_calib)
   tower_M = _block(tower_M)
   tower_M = Dense(50)(tower_M)
   tower_M = _block(tower_M)
   tower_M = Dense(10)(tower_M)
   tower_M = _block(tower_M)
   tower_M = Dense(1)(tower_M)

#   tower_pT_eta_E_M = concatenate( [ tower_pT, tower_eta, tower_E, tower_M ] )

   tower_pT_eta_E = concatenate( [ tower_pT, tower_eta, tower_E ] )
   tower_pT_eta_M = concatenate( [ tower_pT, tower_eta, tower_M ] )
   tower_pT_eta_E_M = concatenate( [ tower_pT_eta_E, tower_pT_eta_M ] )

#   tower_pT_eta_E = concatenate( [ tower_pT, tower_eta, tower_E ] )
#   tower_pT_eta_M = concatenate( [ tower_pT, tower_eta, tower_M ] )
#   tower_pT_E_M   = concatenate( [ tower_pT, tower_E, tower_M ] )
#   tower_pT_eta_E_M = concatenate( [ tower_pT_eta_E, tower_pT_eta_M, tower_pT_E_M ] )
  
   calibrated = Dense( 4, activation='linear', name='calibrated' )(tower_pT_eta_E_M)

   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   adam = Adam( lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001 ) 
   dnn_model.compile( optimizer=adam, loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
   print "INFO: DNN calibration model ResNet compiled"
   return dnn_model

############

BATCH_SIZE = 20000
MAX_EPOCHS = 20
model_filename = "model.calib4.h5" 

callbacks_list = [ 
   # val_loss
#   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, epsilon=0.1, min_lr=0.001, verbose=1), 
   EarlyStopping( monitor='val_loss', patience=5, mode='min', min_delta=0.1, verbose=1 ),
#   ModelCheckpoint( model_filename + "-{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', mode='min', save_best_only=True, verbose=0), 
   ModelCheckpoint( model_filename, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
]

print "INFO: creating calibration DNN"
#dnn = KerasRegressor( build_fn=create_model_calib_4, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4x1, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1, sample_weight=y_weight_mc )
dnn = KerasRegressor( build_fn=create_model_calib_resnet, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1, sample_weight=y_weight_mc )
dnn.fit( X_train_all_encoded, y_train_all )

#dnn.model.save( model_filename )

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

