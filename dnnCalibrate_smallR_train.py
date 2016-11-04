#!/usr/bin/env python

import os, sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import cross_val_score

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

def create_dnn_E():
  model = Sequential()
  model.add( Dense(3, input_dim=3, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(1, init='normal') )
  model.compile( loss='mean_squared_error', optimizer='rmsprop' )
  return model

def create_dnn_pT_E():
  # relu, sigmoid, tanh, linear
  model = Sequential()
  model.add( Dense(3, input_dim=3, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(600, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(2, init='normal') )
  model.compile( loss='mean_squared_error', optimizer='rmsprop' )
  return model

  
def create_dnn_eta_E():
  # relu, sigmoid, tanh, linear
  model = Sequential()
  model.add( Dense(3, input_dim=3, init='normal', activation='linear'))
#  model.add( Dense(400, init='normal', activation='linear'))
#  model.add(Dropout(0.2))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(600, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(2, init='normal') )

#  sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
#  model.compile( loss='mean_squared_error', optimizer=sgd )

#  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#  model.compile( loss='mean_squared_error', optimizer='adam' )
  model.compile( loss='mean_squared_error', optimizer='rmsprop' )


  return model

#################


scaler = StandardScaler() 
#scaler = RobustScaler()

training_filename = sys.argv[1]

#training_dataset = np.loadtxt( training_filename, delimiter=",")
#dataframe = pandas.read_csv( training_filename, header=None )
#training_dataset = dataframe.values
testing_dataset = pd.read_csv( testing_filename, delimiter="," ).values

event_train   = training_dataset[:,:2]
calib_train   = training_dataset[:,2:5]#.astype(float)
nocalib_train = training_dataset[:,5:8]#.astype(float)
truth_train   = training_dataset[:,8:]#.astype(float)

print calib_train
print nocalib_train
print truth_train

calib_train   = scaler.fit_transform( calib_train )
nocalib_train = scaler.transform( nocalib_train )
#truth_train   = scaler.transform( truth_train )
#truth_train = truth_train[:,2:]
truth_train = truth_train[:,::2]

print calib_train
print nocalib_train
print truth_train

dnn = KerasRegressor( build_fn=create_dnn_pT_E, nb_epoch=5, batch_size=5000, verbose=1 )

with open( "scaler.pkl", "wb" ) as file_scaler:
  pickle.dump( scaler, file_scaler )

#################
# Training

#print
#print "INFO: Cross-validation.."
#scores = cross_val_score( dnn, nocalib_train, E_truth_train )
#scores = cross_val_score( dnn, nocalib_train, truth_train )
#print( "INFO: Cross-validation scores: {}".format(scores) )

print
print "INFO: Training.."

#dnn.fit( nocalib_train, E_truth_train )
#dnn.fit( nocalib_train, E_truth_train, callbacks=[early_stopping] )
dnn.fit( nocalib_train, truth_train )

#score_train = dnn.score( nocalib_train, truth_train )
#print
#print "Score (dnn,training):", score_train

dnn.model.save( "dnn.pT_E.h5" )
