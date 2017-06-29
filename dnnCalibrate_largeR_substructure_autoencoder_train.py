#!/usr/bin/env python

import os, sys

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

np.set_printoptions( precision=3, suppress=True, linewidth=120 )

#################

from models import *
from features import *

training_filename = sys.argv[1]

# Set up scalers
#create_scaler = StandardScaler
#create_scaler = MinMaxScaler
create_scaler = RobustScaler
X_scaler = create_scaler()

# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
X_train_all = X_scaler.fit_transform( X_train_all )

# Create autoencoder
n_input_all = len( features_all )
encoding_dim = 15
print "INFO: creating autoencoder %i -> %i -> %i" % ( n_input_all, encoding_dim, n_input_all )

# linear model (=PCA)
encoder_input = Input( shape=(n_input_all,) )
encoded = Dense( encoding_dim )(encoder_input)
decoded = Dense(  n_input_all )(encoded)

#encoder_input = Input( shape=(n_input_all,) )
#encoded = Dense( 20, activation='relu' )(encoder_input)
#encoded = Dense( 16, activation='relu' )(encoded)
#encoded = Dense( encoding_dim, activation='relu' )(encoded)
#decoded = Dense( 16, activation='relu' )(encoded)
#decoded = Dense( 20, activation='relu' )(decoded)
#decoded = Dense(    n_input_all )(decoded)

autoencoder = Model( inputs=encoder_input, outputs=decoded)
#autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.compile( optimizer = 'adam', loss='mean_squared_error' )

encoder_filename = "encoder.h5"
callbacks_list = [
   # val_loss
   ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001, verbose=1),
   EarlyStopping( monitor='loss', patience=3, mode='min', verbose=1 ),
]
autoencoder.fit( X_train_all, X_train_all, epochs=10, batch_size=200, validation_split=0.05, callbacks=callbacks_list, verbose=1 )

print "INFO: Auto-encoder fit finished"

# now create encoder-only model
encoder = Model(inputs=encoder_input, outputs=encoded)
encoder.encoding_dim = encoding_dim
encoder.compile( optimizer = 'adam', loss='mean_squared_error' )
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

