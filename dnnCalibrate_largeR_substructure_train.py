#!/usr/bin/env python

import os, sys

from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

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

#################

def create_model():
   model_pT = Sequential()
   model_pT.add( Dense( n_input_pT, input_dim=n_input_pT ) )
   model_pT.add( Dense(50, activation='linear'))
   model_pT.add( Dense(20, activation='linear'))
   model_pT.add( Dense(1, activation='linear'))

   model_eta = Sequential()
   model_eta.add( Dense( n_input_eta, input_dim=n_input_eta ) )
   model_eta.add( Dense(50, activation='linear'))
   model_eta.add( Dense(20, activation='linear'))
   model_eta.add( Dense(1, activation='linear'))

   model_E = Sequential()
   model_E.add( Dense( n_input_E, input_dim=n_input_E ) )
   model_E.add( Dense(50, activation='linear'))
   model_E.add( Dense(20, activation='linear'))
   model_E.add( Dense(1, activation='linear'))

   model_M = Sequential()
   model_M.add( Dense( n_input_M, input_dim=n_input_M ) )
   model_M.add( Dense(50, activation='linear'))
   model_M.add( Dense(20, activation='linear'))
   model_M.add( Dense(1, activation='linear'))

   merged = Merge( [ model_pT, model_eta, model_E, model_M ], mode='concat' )

   model = Sequential()
   model.add(merged)
   model.add( Dense( 4, activation='linear' ) )
   model.compile( loss='mean_squared_error', optimizer='adam' )

   return model

#################

training_filename = sys.argv[1]
header = [
  "eventNumber", "weight", "mu", "prw", 
  "fjet1_truth_pt", "fjet1_truth_eta", "fjet1_truth_E", "fjet1_truth_P", "fjet1_truth_M",
  "fjet1_nocalib_pt", "fjet1_nocalib_eta", "fjet1_nocalib_E", "fjet1_nocalib_P", "fjet1_nocalib_M",
  "fjet1_Nconstit", "fjet1_untrimNtrk500", 
  "fjet1_D2", "fjet1_C2", 
  "fjet1_Tau1", "fjet1_Tau1_wta",
  "fjet1_Tau2", "fjet1_Tau2_wta",
  "fjet1_Tau3", "fjet1_Tau3_wta",
  "fjet1_Angularity", "fjet1_Aplanarity", "fjet1_PlanarFlow", "fjet1_Sphericity", "fjet1_ThrustMaj", "fjet1_ThrustMin",
  "fjet1_KtDR", "fjet1_Mu12", "fjet1_Width", "fjet1_Qw", "fjet1_Split12", "fjet1_Split23", "fjet1_Split34",
  "fjet1_calib_pt", "fjet1_calib_eta", "fjet1_calib_E", "fjet1_calib_P", "fjet1_calib_M",
  "alpha_pt", "alpha_eta", "alpha_E", "alpha_M"
]

# Set up scalers
scaler = MinMaxScaler()
poly = PolynomialFeatures(2)

# read in input file
#training_dataset = pd.read_csv( training_filename, delimiter="," ).values
df_training = pd.read_csv( training_filename, delimiter=',', names=header )
#print df_training

# transverse momentum
features_pT = [ "fjet1_nocalib_pt", "fjet1_nocalib_eta", "fjet1_nocalib_E", "fjet1_nocalib_P", "fjet1_nocalib_M",
                "fjet1_D2", "fjet1_C2", "fjet1_Tau1", "fjet1_Tau2", "fjet1_Tau3", "fjet1_Split12", "fjet1_Split23", "fjet1_Split34" ] 
X_train_pT = df_training[features_pT].values

# (pseudo)rapidity
features_eta = [ "fjet1_nocalib_pt", "fjet1_nocalib_eta", "fjet1_nocalib_E", "fjet1_nocalib_P", "fjet1_nocalib_M",
                 "fjet1_D2", "fjet1_C2", "fjet1_Tau1", "fjet1_Tau2", "fjet1_Tau3", "fjet1_Split12", "fjet1_Split23", "fjet1_Split34" ]
X_train_eta = df_training[features_eta].values

# energy
features_E  = [ "fjet1_nocalib_pt", "fjet1_nocalib_eta", "fjet1_nocalib_E", "fjet1_nocalib_P", "fjet1_nocalib_M",
                "fjet1_D2", "fjet1_C2", "fjet1_Tau1", "fjet1_Tau2", "fjet1_Tau3", "fjet1_Split12", "fjet1_Split23", "fjet1_Split34" ]
X_train_E = df_training[features_E].values

# mass
features_M  = [ "fjet1_nocalib_pt", "fjet1_nocalib_eta", "fjet1_nocalib_E", "fjet1_nocalib_P", "fjet1_nocalib_M",
                "fjet1_D2", "fjet1_C2", "fjet1_Tau1", "fjet1_Tau2", "fjet1_Tau3", "fjet1_Split12", "fjet1_Split23", "fjet1_Split34" ]
X_train_M = df_training[features_M].values

y_train = df_training[ [ "alpha_pt", "alpha_eta", "alpha_E", "alpha_M" ] ].values

n_input_pT  = len( features_pT )
n_input_eta = len( features_eta )
n_input_E   = len( features_E )
n_input_M   = len( features_M )

print "INFO: N inputs pT: ", n_input_pT
print "INFO: N inputs eta:", n_input_eta
print "INFO: N inputs E:  ", n_input_E
print "INFO: N inputs M:  ", n_input_M

dnn = KerasRegressor( build_fn=create_model, nb_epoch=50, batch_size=100, verbose=1 )

dnn.fit( [ X_train_pT, X_train_eta, X_train_E, X_train_M ], y_train )

dnn.model.save( "dnn.largeR_substructure.h5" )

