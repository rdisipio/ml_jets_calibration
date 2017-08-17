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

def custom_loss(y_true, y_pred):
#   y_t = K.clip( y_true, K.epsilon(), 1. )
#   y_p = K.clip( y_pred, K.epsilon(), 1. )
#   return K.mean( K.abs( y_p/y_t - 1 ) )
#    return K.mean(K.square((y_pred/y_true) - 1))
#   return K.mean( (y_pred/y_true) - 1. )
   return K.mean( (y_pred - y_true)/K.sqrt(y_true) )

#~~~~~~~~~

def create_model_calib_4x1_merged():
#   input_calib = Input( shape=(encoding_dim, ))
#, kernel_regularizer=l2(0.01)
 
   #kreg = None
   kreg = kernel_regularizer=l2(0.01) 
 
   def _block_highway(_input):
      n = K.int_shape(_input)[1]
      _H = Dense(n)(_input)
      _H = Activation('relu')(_H)

      _T = Dense(n)(_input)
      _T = Activation('sigmoid')(_T)

      _output = multiply( [ _H, _T ] )
      _output = add( [ _output, _input ] )

      _C = multiply( [ _input, _T ] )
      _output = subtract( [ _output, _C ] )

      return _output

   def _block_resnet(_input, activation='tanh'):
      n = K.int_shape(_input)[1]
      _output = Dense(n, kernel_regularizer=kreg)(_input)
      #_output = BatchNormalization()(_output)
      _output = Activation(activation)(_output)
      _output = Dense(n, kernel_regularizer=kreg)(_output)
      #_output = subtract( [ _output, _input ] )
      _output = add( [ _output, _input ] )
      _output = Activation(activation)(_output)
      return _output

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   x_pT   = Dense(n_input_pT)(input_pT)
   x_eta  = Dense(n_input_eta)(input_eta)
   x_E    = Dense(n_input_E)(input_E)
   x_M    = Dense(n_input_M)(input_M)

   x_pT  = Dense(300)(x_pT)
   x_pT  = _block_resnet(x_pT)
   x_pT  = Dense(100)(x_pT)
   x_pT  = _block_resnet(x_pT)
   x_pT  = Dense(50)(x_pT)
   x_pT  = _block_resnet(x_pT)
   x_pT  = Dense(1)(x_pT)

   x_eta = Dense( 10, activation='tanh', kernel_regularizer=kreg )(x_eta)
   x_eta = Dense(5, activation='tanh', kernel_regularizer=kreg )(x_eta)
   x_eta = Dense(1)(x_eta)

   x_E  = Dense(300)(x_E)
   x_E  = _block_resnet(x_E)
   x_E  = Dense(100)(x_E)
   x_E  = _block_resnet(x_E)
   x_E  = Dense(50)(x_E)
   x_E  = _block_resnet(x_E)
   x_E  = Dense(1)(x_E)

   x_M  = Dense(500)(x_M)
   x_M  = _block_resnet(x_M, "tanh")
   x_M  = Dense(300)(x_M)
   x_M  = _block_resnet(x_M, "tanh")
   x_M  = Dense(100)(x_M)
   x_M  = _block_resnet(x_M, "tanh")
   x_M  = Dense(1)(x_M)

#   x = concatenate( [ x_pT, x_E, x_M ] )
#   x = Dense(3)(x)

   x_pT_eta_E = concatenate( [ x_pT, x_eta, x_E ] )
   x_pT_eta_M = concatenate( [ x_pT, x_eta, x_M ] )
   dnn_calib = concatenate( [ x_pT_eta_E, x_pT_eta_M ] )
   dnn_calib = Dense(6, activation='linear')(dnn_calib)
   dnn_calib = Dense(5, activation='linear')(dnn_calib)
   dnn_calib = Dense(4, activation='linear')(dnn_calib)

#   x_pT_E = concatenate( [ x_pT, x_E ] )
#   x_pT_E = Dense( 300, activation='relu' )(x_pT_E)
#   x_pT_E = Dense( 200, activation='relu' )(x_pT_E)
#   x_pT_E = Dense( 100, activation='relu' )(x_pT_E)
#   x_pT_E = Dense(  50, activation='relu' )(x_pT_E)
#   x_pT_E = Dense(1)(x_pT_E)

#   x_pT_E_M = concatenate( [ x_pT_E, x_pT_M ] )
#   x_pT_E_M = Dense(2)(x_pT_E_M)

#   dnn_calib = concatenate( [ x_eta, x_pT_M, x_pT_E, ] )
#   dnn_calib = concatenate( [ x_eta, x_pT_E_M ] )
#   dnn_calib =  concatenate( [ x_pT, x_eta, x_E, x_M ] ) 
#   dnn_calib =  concatenate( [ x_eta, x ] )


   dnn_calib   = Dense(4)(dnn_calib)
   dnn_model   = Model( inputs=[input_pT,input_eta,input_E,input_M], outputs=dnn_calib )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
#   dnn_model.compile( optimizer='adam', loss=custom_loss )
   print "INFO: DNN calibration model 4x1 merged compiled"
   return dnn_model

###############

def create_model_calib_4():
#   input_calib = Input( shape=(encoding_dim, ))

   kreg = None
   #kreg = kernel_regularizer=l2(0.01)

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   x_pT   = Dense(n_input_pT)(input_pT)
   x_eta  = Dense(n_input_eta)(input_eta)
   x_E    = Dense(n_input_E)(input_E)
   x_M    = Dense(n_input_M)(input_M)

   input_calib = concatenate( [ x_pT, x_eta, x_E, x_M ] )

# , kernel_regularizer=l2(0.1)
   dnn_calib   = Dense( 500, activation='relu' )(input_calib)
   dnn_calib   = Dense( 300, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense( 200, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense( 100, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense(  50, activation='relu', kernel_regularizer=kreg )(dnn_calib)
   dnn_calib   = Dense(   4 )(dnn_calib)
   dnn_model   = Model( inputs=[input_pT,input_eta,input_E,input_M], outputs=dnn_calib )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
#   dnn_model.compile( optimizer='adam', loss=custom_loss )
   print "INFO: DNN calibration model compiled"
   return dnn_model

#~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_model_calib_4x1():
#   input_calib = Input( shape=(encoding_dim, ))

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

#   x_pT_eta_E_M = Dense(500, activation='relu')(input_calib)
#   x_pT_eta_E_M = Dense(300, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense(100, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense( 50, activation='relu')(x_pT_eta_E_M)
#   x_pT_eta_E_M = Dense(  4)(x_pT_eta_E_M)
   
   dnn_calib_pT = Dense( 500 )(input_pT)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense( 300)(dnn_calib_pT) 
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense( 100)(dnn_calib_pT)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense(  50)(dnn_calib_pT)
   dnn_calib_pT = Activation('relu')(dnn_calib_pT)
   dnn_calib_pT = Dense(1)(dnn_calib_pT)

   dnn_calib_eta = Dense( 500 )(input_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense( 300)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense( 100)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense(  50)(dnn_calib_eta)
   dnn_calib_eta = Activation('relu')(dnn_calib_eta)
   dnn_calib_eta = Dense(1)(dnn_calib_eta)

   dnn_calib_E = Dense( 500 )(input_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense( 300)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense( 100)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense(  50)(dnn_calib_E)
   dnn_calib_E = Activation('relu')(dnn_calib_E)
   dnn_calib_E = Dense(1)(dnn_calib_E)

   dnn_calib_M = Dense( 500 )(input_M)
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
#   dnn_calib_pT_eta_E_M = Dense(4, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(200, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(100, activation='linear')(dnn_calib_pT_eta_E_M)
#   dnn_calib_pT_eta_E_M = Dense(4)(dnn_calib_pT_eta_E_M)

 #  dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT, dnn_calib_eta, dnn_calib_E, dnn_calib_M ] )
#   dnn_calib_pT_eta_E_M = concatenate( [ dnn_calib_pT_eta_E, dnn_calib_M ] )

#   dnn_calib_pT_eta_E_M = average( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )
#   dnn_calib_pT_eta_E_M = maximum( [ dnn_calib_pT_eta_E, dnn_calib_pT_eta_M ] )
   
#   calibrated = average( [ dnn_calib_pT_eta_E_M, x_pT_eta_E_M ] )

   calibrated = Dense( 4, activation='linear', name='calibrated' )(dnn_calib_pT_eta_E_M)

   input_calib = [ input_pT, input_eta, input_E, input_M ]
   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
   print "INFO: DNN calibration model compiled"
   return dnn_model

def create_model_calib_resnet():
   def _block_resnet(_input):
      n = K.int_shape(_input)[1]
      _output = Dense(n)(_input)
      #_output = BatchNormalization()(_output)
      _output = Activation('relu')(_output)
      _output = Dense(n)(_output)
      _output = add( [ _output, _input ] )
      _output = Activation('relu')(_output)
      return _output
   
#   input_calib = Input( shape=(encoding_dim, ))

   input_pT  = Input( shape=(n_input_pT,) )
   input_eta = Input( shape=(n_input_eta,) )
   input_E   = Input( shape=(n_input_E,) )
   input_M   = Input( shape=(n_input_M,) )

   tower_pT = Dense(50, kernel_initializer='glorot_normal' )(input_pT)
   for n_nodes in [ 30, 20 ]:
      tower_pT = Dense(n_nodes)(tower_pT)
      tower_pT = _block_resnet(tower_pT)
   tower_pT = Dense(1)(tower_pT)
   
   tower_eta = Dense(10, kernel_initializer='glorot_normal' )(input_eta)
   for n_nodes in [ 5 ]:
      tower_eta = Dense(n_nodes)(tower_eta)
      tower_eta = _block_resnet(tower_eta)
   tower_eta = Dense(1)(tower_eta)

   tower_E = Dense(50, kernel_initializer='glorot_normal' )(input_E)
   for n_nodes in [ 30, 20 ]:
      tower_E = Dense(n_nodes)(tower_E)
      tower_E = _block_resnet(tower_E)
   tower_E = Dense(1)(tower_E)

   tower_M = Dense(100, kernel_initializer='glorot_normal' )(input_M)
   for n_nodes in [ 50, 30 ]:
      tower_M = Dense(n_nodes)(tower_M)
      tower_M = _block_resnet(tower_M)
   tower_M = Dense(1)(tower_M)

#   tower_pT_eta_E   = concatenate( [ tower_pT, tower_eta, tower_E ] )
#   tower_pT_eta_M   = concatenate( [ tower_pT, tower_eta, tower_M ] )
#   tower_pT_eta_E_M = concatenate( [ tower_pT_eta_E, tower_pT_eta_M ] )

   tower_pT_eta_E_M = concatenate( [ tower_pT, tower_eta, tower_E, tower_M ] )  
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)
   tower_pT_eta_E_M = Dense(4)(tower_pT_eta_E_M)

#   x_pT_E = concatenate( [  tower_pT, tower_E ] )
#   x_pT_M = concatenate( [  tower_pT, tower_M ] )
#   x_pT_E_M = concatenate( [ x_pT_E, x_pT_M ] )
#   tower_pT_eta_E_M = concatenate( [ tower_eta, x_pT_E_M ] )

   calibrated = Dense( 4, activation='linear', name='calibrated' )(tower_pT_eta_E_M)
   input_calib = [ input_pT, input_eta, input_E, input_M ] 
   dnn_model  = Model( inputs=input_calib, outputs=calibrated )

#   adam = Adam( lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001 ) 
   dnn_model.compile( optimizer='adam', loss='mean_squared_error' )
#   dnn_model.compile( optimizer='adam', loss='mean_absolute_error' )
   print "INFO: DNN calibration model ResNet compiled"
   return dnn_model

############

BATCH_SIZE = 500
MAX_EPOCHS = 20
model_filename = "model.calib4.h5" 

callbacks_list = [ 
   # val_loss
#   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, epsilon=0.1, min_lr=0.001, verbose=1), 
#   EarlyStopping( monitor='val_loss', patience=5, mode='min', min_delta=0.0005, verbose=1 ),
#   ModelCheckpoint( model_filename + "-{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', mode='min', save_best_only=True, verbose=0), 
   ModelCheckpoint( model_filename, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
]

print "INFO: creating calibration DNN"
#dnn = KerasRegressor( build_fn=create_model_calib_4, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 )#, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4x1, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 ) #, sample_weight=y_weight_mc )
dnn = KerasRegressor( build_fn=create_model_calib_resnet, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 )#, sample_weight=y_weight_mc )
#dnn = KerasRegressor( build_fn=create_model_calib_4x1_merged, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=callbacks_list, verbose=1 ) #, sample_weight=y_weight_mc )
dnn.fit( [X_train_pT,X_train_eta,X_train_E,X_train_M], y_train_all )

#dnn.model.save( model_filename )

scaler_filename = "scaler.largeR_substructure.pkl"
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

