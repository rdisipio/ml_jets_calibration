#!/usr/bin/env python

import os, sys

from keras.models import Sequential

from keras.layers import Merge
from keras.layers import Dense, Activation
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers.advanced_activations import ELU

from keras.wrappers.scikit_learn import KerasRegressor

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

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

checkpoint = ModelCheckpoint( "checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#################

def eta_range( y ):
   ymax = 2.5
   if abs(ymax) <  abs(y): return 0.
   return y

#   model_pT.add(Dropout(0.2))

def create_pT():
   model_pT = Sequential()

   model_pT.add( Dense( 300, input_dim=n_input_pT) )
   model_pT.add( ELU() )

#   model_pT.add(Dropout(0.1))

   model_pT.add( Dense(  1, init='uniform'))
   model_pT.add( ELU() )

   return model_pT

def create_eta():
   model_eta = Sequential()

   model_eta.add( Dense( 300, input_dim=n_input_eta) )

   model_eta.add( Dense(1) )

   return model_eta

def create_E():
   model_E = Sequential()

   model_E.add( Dense( 300, input_dim=n_input_E) )
   model_E.add( ELU() )

#   model_E.add(Dropout(0.1))

   model_E.add( Dense(1) )
   model_E.add( ELU() )

   return model_E

def create_M():
   model_M = Sequential()

   model_M.add( Dense( 400, input_dim=n_input_M) )
   model_M.add( ELU() )

#   model_M.add(Dropout(0.1))

   model_M.add( Dense(1) )
   model_M.add( ELU() )

   return model_M

def create_model():
   model_pT  = create_pT()
   model_eta = create_eta()
   model_E   = create_E()
   model_M   = create_M()

   merged = Merge( [ model_pT, model_eta, model_E, model_M ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 4 ) )

   model.compile( loss='mean_squared_error', optimizer='adam' )
#   model.compile( loss='mean_squared_logarithmic_error', optimizer='adam' )

   return model

#################

training_filename = sys.argv[1]

# Sana's
header = [ 
 "jet_Weight", 
 "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_P", "jet_truth_M",
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 "jet_Nconstit",
 "jet_D2", "jet_C2",
 "jet_Tau1_wta", "jet_Tau2_wta", "jet_Tau3_wta", "jet_Tau21_wta", "jet_Tau32_wta",
 "jet_Angularity", "jet_Aplanarity", "jet_PlanarFlow", "jet_Sphericity",
 "jet_Width",
# "jet_ECF1", "jet_ECF2", "jet_ECF3",
 "jet_calib_Pt", "jet_calib_Eta", "jet_calib_E", "jet_calib_P", "jet_calib_M",
]  

# Set up scalers
scaler_pT  = MinMaxScaler()
scaler_eta = MinMaxScaler()
scaler_E   = MinMaxScaler()
scaler_M   = MinMaxScaler() 
#poly = PolynomialFeatures(2)

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

# transverse momentum
features_pT = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
  ] 
X_train_pT = df_training[features_pT].values
X_train_pT = scaler_pT.fit_transform( X_train_pT )

# (pseudo)rapidity
features_eta = [
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
]
X_train_eta = df_training[features_eta].values
X_train_eta = scaler_eta.fit_transform( X_train_eta )

# energy
features_E  = [ 
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 ]
X_train_E = df_training[features_E].values
X_train_E = scaler_E.fit_transform( X_train_E )

# mass
features_M  = [ 
 "jet_nocalib_Pt", "jet_nocalib_Eta", "jet_nocalib_E", "jet_nocalib_P", "jet_nocalib_M",
 ]
X_train_M = df_training[features_M].values
X_train_M = scaler_M.fit_transform( X_train_M )

y_train = df_training[ [ "jet_truth_Pt", "jet_truth_Eta", "jet_truth_E", "jet_truth_M" ] ].values

n_input_pT  = len( features_pT )
n_input_eta = len( features_eta )
n_input_E   = len( features_E )
n_input_M   = len( features_M )

print "INFO: N inputs pT: ", n_input_pT
print "INFO: N inputs eta:", n_input_eta
print "INFO: N inputs E:  ", n_input_E
print "INFO: N inputs M:  ", n_input_M

dnn = KerasRegressor( build_fn=create_model, nb_epoch=20, batch_size=1000, verbose=1 )

history = dnn.fit( [ X_train_pT, X_train_eta, X_train_E, X_train_M ], y_train ) #, callbacks=callbacks_list )

res = dnn.score( [ X_train_pT, X_train_eta, X_train_E, X_train_M ], y_train )
print "Score(training):", res

dnn.model.save( "dnn.largeR_substructure.sketabch.h5" )

with open( "scaler.largeR_substructure.pkl", "wb" ) as file_scaler:
  pickle.dump( scaler_pT,  file_scaler )
  pickle.dump( scaler_eta, file_scaler )
  pickle.dump( scaler_E,   file_scaler )
  pickle.dump( scaler_M,   file_scaler )

#  pickle.dump( poly,   file_scaler )
print "INFO: scaler saved in file", "scaler.largeR_substructure.sketabch.pkl"

