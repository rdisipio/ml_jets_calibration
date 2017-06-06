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
create_scaler = StandardScaler
#create_scaler = MinMaxScaler
X_scaler = create_scaler()

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
X_train_all = X_scaler.fit_transform( X_train_all )

# Create autoencoder
n_input_all = len( features_all )
encoding_dim = 12
print "INFO: creating autoencoder %i -> %i ->%i" % ( n_input_all, encoding_dim, n_input_all )

encoder_input = Input( shape=(n_input_all,) )

# linear model (=PCA)
#encoded = Dense( encoding_dim )(encoder_input)
#decoded = Dense(  n_input_all )(encoded)

encoded = Dense( 30, activation='tanh' )(encoder_input)
encoded = Dense( 20, activation='tanh' )(encoded)
encoded = Dense( encoding_dim, activation='tanh' )(encoded)

decoded = Dense( 20, activation='tanh' )(encoded)
decoded = Dense( 30, activation='tanh' )(decoded)
decoded = Dense(    n_input_all )(decoded)

autoencoder = Model( inputs=encoder_input, outputs=decoded)
#autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.compile( optimizer = 'adam', loss='mean_squared_error' )
autoencoder.fit( X_train_all, X_train_all, epochs=10, batch_size=1000, validation_split=0.05, callbacks=callbacks_list, verbose=1 )

print "INFO: Auto-encoder fit finished"

# now create encoder-only model
encoder = Model(inputs=encoder_input, outputs=encoded)
encoder.encoding_dim = encoding_dim
#encode.compile( optimizer = 'adam', loss='mean_squared_error' )
encoder_filename = "encoder.h5"
encoder.save( encoder_filename )
print "INFO: encoder save to file:", encoder_filename

scaler_filename = "X_scaler.pkl"
with open( scaler_filename, "wb" ) as file_scaler:
  pickle.dump( X_scaler,     file_scaler )
print "INFO: X_scaler saved to file:", scaler_filename

# these are the compressed data
X_train_all_encoded = encoder.predict(X_train_all)
print "INFO: example of compressed data:"
print X_train_all_encoded
