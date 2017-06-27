#!/usr/bin/env python

import sys, os
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np
import pandas as pd

from math import sqrt, pow

from models import *
from features import *

from ROOT import *

training_filename = sys.argv[1]

# Set up scalers
filename_scaler = "X_scaler.pkl"
with open( filename_scaler, "rb" ) as file_scaler:
   X_scaler = pickle.load( file_scaler )

print "INFO: X_scaler loaded from file", filename_scaler


# read in input file
df_training = pd.read_csv( training_filename, delimiter=',', names=header )

X_train_all = df_training[features_all].values
X_train_all = X_scaler.transform( X_train_all )

# Create autoencoder
n_input_all = len( features_all )
encoder = load_model( "encoder.h5" )
encoding_dim = encoder.layers[-1].output_shape[1]
print "INFO: loaded encoder %i -> %i" % ( n_input_all, encoding_dim )

# these are the compressed data
X_train_all_encoded = encoder.predict(X_train_all)
print "INFO: example of compressed data:"
print X_train_all_encoded


f = TFile.Open( "autoencoder.root", "RECREATE" )
histograms = [ TH1F( "variable_%i"%i, "Variable %i"%i, 100, -1., 1. ) for i in range(encoding_dim) ]

principal = TPrincipal( encoding_dim, "ND" )

n_entries = len(X_train_all_encoded)
print "INFO: filling histograms with %i entries" % n_entries
for ientry in range(n_entries):
   if ( n_entries < 10 ) or ( (ientry+1) % int(float(n_entries)/10.)  == 0 ):
     perc = 100. * ientry / float(n_entries)
     print "INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc )

   for iobs in range(encoding_dim):
      v = X_train_all_encoded[ientry][iobs]
      histograms[iobs].Fill( v )
   
   row = np.asarray(X_train_all_encoded[ientry], dtype=np.float64)
   principal.AddRow(row)

principal.MakePrincipals()
cov = principal.GetCovarianceMatrix()

# convert to TH2D
h2_cov = TH2D( "covariance", "Covariance matrix (Absolute cross-sections)", encoding_dim, 0.5, encoding_dim+0.5, encoding_dim, 0.5, encoding_dim+0.5 )
for i in range(encoding_dim):
  for j in range(encoding_dim):
      h2_cov.SetBinContent( i+1, j+1, cov(i,j) )

h2_corr = TH2D( "correlation", "Correlation matrix (Absolute cross-sections)", encoding_dim, 0.5, encoding_dim+0.5, encoding_dim, 0.5, encoding_dim+0.5 )
for i in range(encoding_dim):
  sigma_i = h2_cov.GetBinContent( i+1, i+1 )
  for j in range(encoding_dim):
    sigma_j = h2_cov.GetBinContent( j+1, j+1 )

    Cij = h2_cov.GetBinContent( i+1, j+1 )
    cij = Cij / sqrt( sigma_i * sigma_j )
    h2_corr.SetBinContent( i+1, j+1, cij )
h2_corr.SetMaximum(  1.0 )
h2_corr.SetMinimum( -1.0 )

for h in histograms: h.Write()
h2_cov.Write( "covariance" )
h2_corr.Write( "correlation" )

f.Close()


