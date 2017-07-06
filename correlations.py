#!/usr/bin/env python

import os, sys

import numpy as np
import pandas as pd

from features import *

from ROOT import *
from math import pow, sqrt

np.set_printoptions( precision=3, suppress=True, linewidth=120 )

training_filename = sys.argv[1]

df_training = pd.read_csv( training_filename, delimiter=',', names=header )
X_train_all = df_training[features_all].values

print X_train_all

n_features = len( features_all )
n_entries  = len( X_train_all )

principal = TPrincipal( n_features )
print "INFO: looping over %i entries" % n_entries
for i in range( n_entries ):

  if ( n_entries < 10 ) or ( (i+1) % int(float(n_entries)/10.)  == 0 ):
    perc = 100. * i / float(n_entries)
    print "INFO: Event %-9i  (%3.0f %%)" % ( i, perc )

  a = np.array( X_train_all[i], dtype='float64' )
  principal.AddRow( a )

principal.MakePrincipals()
cov = principal.GetCovarianceMatrix()

# convert to TH2D
h2_cov = TH2D( "h2_covariance", "Covariance matrix (Absolute cross-sections)", n_features, 0.5, n_features+0.5, n_features, 0.5, n_features+0.5 )
h2_cov.GetXaxis().SetLabelSize(0.02)
h2_cov.GetYaxis().SetLabelSize(0.02)
for i in range(n_features):
  for j in range(n_features):
      h2_cov.SetBinContent( i+1, j+1, cov(i,j) )

h2_corr = TH2D( "h2_correlation", "Correlation matrix (Absolute cross-sections)", n_features, 0.5, n_features+0.5, n_features, 0.5, n_features+0.5 )
h2_corr.GetXaxis().SetLabelSize(0.02)
h2_corr.GetYaxis().SetLabelSize(0.02)
for i in range(n_features):
  sigma_i = h2_cov.GetBinContent( i+1, i+1 )
  for j in range(n_features):
    sigma_j = h2_cov.GetBinContent( j+1, j+1 )

    Cij = h2_cov.GetBinContent( i+1, j+1 )
    cij = Cij / sqrt( sigma_i * sigma_j )
    h2_corr.SetBinContent( i+1, j+1, cij )
h2_corr.SetMaximum(  1.0 )
h2_corr.SetMinimum( -1.0 )

for k in range(n_features):
  label = features_all[k]
  h2_cov.GetXaxis().SetBinLabel( k+1, label )
  h2_cov.GetYaxis().SetBinLabel( k+1, label )
  h2_corr.GetXaxis().SetBinLabel( k+1, label )
  h2_corr.GetYaxis().SetBinLabel( k+1, label )

ofile = TFile.Open( "covariance.root", "RECREATE" )
cov.Write( "m_covariance" )
h2_cov.Write()
h2_corr.Write()
ofile.Close()
print "INFO: output file created", ofile.GetName()
