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

def create_dnn_E():
  model = Sequential()
  model.add( Dense(3, input_dim=n_inputs, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(1, init='normal') )
  model.compile( loss='mean_squared_error', optimizer='rmsprop' )
  return model

#~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dnn_eta():
  model = Sequential()
  model.add( Dense(3, input_dim=n_inputs, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
#  model.add(Dropout(0.2))
  model.add( Dense(1, init='normal') )
  model.compile( loss='mean_squared_error', optimizer='adam' )
  return model

#~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dnn_pT_E():
  # relu, sigmoid, tanh, linear
  model = Sequential()
  model.add( Dense(3, input_dim=n_inputs, init='normal', activation='linear'))

#  model.add( Dense(300, init='normal', activation='linear'))
#  model.add(Dropout(0.2))

  model.add( Dense(600, init='normal', activation='linear'))
  model.add(Dropout(0.2))

  model.add( Dense(1200, init='normal', activation='linear'))
  model.add(Dropout(0.2))

  model.add( Dense(2, init='normal') )

  model.compile( loss='mean_squared_error', optimizer='adam' )
  return model


#~~~~~~~~~~~~~~~~~~~~~~~~~
  
def create_dnn_eta_E():
  # relu, sigmoid, tanh, linear
  model = Sequential()
  model.add( Dense(3, input_dim=n_inputs, init='normal', activation='linear'))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(300, init='normal', activation='linear'))
  model.add(Dropout(0.2))
  model.add( Dense(2, init='normal') )
  model.compile( loss='mean_squared_error', optimizer='adam' )
#  model.compile( loss='mean_squared_error', optimizer='rmsprop' )
  return model

#~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dnn_pT_eta_E():
  model = Sequential()

  model.add( Dense(3, input_dim=n_inputs, init='normal', activation='linear'))

  model.add( Dense(300, init='normal', activation='relu'))

  model.add( Dense(300, init='normal', activation='relu'))

  model.add( Dense(200, init='normal', activation='relu'))

  model.add( Dense(3, init='normal') )

  model.compile( loss='mean_squared_error', optimizer='adam' )

  return model


#################


calibration = "pT_eta_E"
#calibration = "pT_E"
#calibration = "eta"

scaler = StandardScaler() 
#scaler = RobustScaler()
poly = PolynomialFeatures(2)

training_filename = sys.argv[1]

#training_dataset = np.loadtxt( training_filename, delimiter=",")
#dataframe = pandas.read_csv( training_filename, header=None )
#training_dataset = dataframe.values

training_dataset = pd.read_csv( training_filename, delimiter="," ).values

#training_dataset = sklearn.utils.shuffle( training_dataset )

# load four-vectors in (pT,eta,phi,E) representation
event_train   = training_dataset[:,:3]
calib_train   = training_dataset[:,3:8]
nocalib_train = training_dataset[:,8:13]
truth_train   = training_dataset[:,13:]

print "INFO: training calib:"
print calib_train
print "INFO: training nocalib:"
print nocalib_train
print "INFO: training truth:"
print truth_train

#nocalib_train = poly.fit_transform( nocalib_train )
#nocalib_train = scaler.fit_transform( nocalib_train )

with open( "scaler.smallR.%s.pkl" % calibration, "wb" ) as file_scaler:
  pickle.dump( scaler, file_scaler )
  pickle.dump( poly,   file_scaler )

x_train = nocalib_train

if calibration == "pT":
   y_train = truth_train[:,0]
if calibration == "eta":
   y_train = truth_train[:,1]
if calibration == "E":
   y_train = truth_train[:,2]
if calibration == "eta_E":
   y_train = truth_train[:,1:] #(eta,E)
if calibration == "pT_E":
   y_train = truth_train[:,::2] #(pT,E)
if calibration == "pT_eta_E":
   y_train = truth_train[:,:3] #(pT,eta,E)

n_inputs = len( x_train[0] )

print "INFO: x train (transformed):"
print nocalib_train
print "INFO: y train:"
print y_train

if calibration == "pT_E":
   dnn = KerasRegressor( build_fn=create_dnn_pT_E, nb_epoch=5, batch_size=5000, verbose=1 )
elif calibration == "eta_E":
   dnn = KerasRegressor( build_fn=create_dnn_eta_E, nb_epoch=5, batch_size=5000, verbose=1 )
elif calibration == "pT_eta_E":
   dnn = KerasRegressor( build_fn=create_dnn_pT_eta_E, nb_epoch=5, batch_size=5000, verbose=1 )
elif calibration == "eta":
   dnn = KerasRegressor( build_fn=create_dnn_eta, nb_epoch=5, batch_size=1000, verbose=1 )
else:
   print "ERROR: unknown calibration scheme", calibration

print "INFO: calibration scheme", calibration

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
dnn.fit( x_train, y_train )

score_train = dnn.score( x_train, y_train )
#print
print "Score (dnn,training):", score_train

dnn.model.save( "dnn.smallR.%s.h5" % calibration )

print "INFO: model summary:"
dnn.model.summary()

print "INFO: saved file", "dnn.smallR.%s.h5" % calibration
